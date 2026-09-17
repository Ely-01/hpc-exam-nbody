/*
 * benchmark_rsqrt_kernel.c
 *
 * Microbenchmark for the reciprocal-square-root optimisation discussed in the
 * exam assignment. This does not integrate the N-body system; it isolates the
 * inverse-distance operation so that the scalar libm path can be compared with
 * a SIMD rsqrt estimate plus Newton refinements.
 */

#define _POSIX_C_SOURCE 200809L

#include "nbody_core.h"

#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum rsqrt_kernel_mode_e
{
  RSQRT_KERNEL_SQRTF,
  RSQRT_KERNEL_RSQRT0,
  RSQRT_KERNEL_RSQRT1,
  RSQRT_KERNEL_RSQRT2
} rsqrt_kernel_mode_t;

static rsqrt_kernel_mode_t parse_mode (const char *text)
{
  if (strcmp (text, "sqrtf") == 0)
    return RSQRT_KERNEL_SQRTF;
  if (strcmp (text, "rsqrt0") == 0)
    return RSQRT_KERNEL_RSQRT0;
  if (strcmp (text, "rsqrt1") == 0)
    return RSQRT_KERNEL_RSQRT1;
  if (strcmp (text, "rsqrt2") == 0)
    return RSQRT_KERNEL_RSQRT2;

  nbody_die ("invalid mode '%s': expected sqrtf, rsqrt0, rsqrt1, or rsqrt2",
              text);
  return RSQRT_KERNEL_SQRTF;
}

static const char *mode_name (rsqrt_kernel_mode_t mode)
{
  switch (mode)
    {
    case RSQRT_KERNEL_SQRTF:
      return "sqrtf";
    case RSQRT_KERNEL_RSQRT0:
      return "rsqrt0";
    case RSQRT_KERNEL_RSQRT1:
      return "rsqrt1";
    case RSQRT_KERNEL_RSQRT2:
      return "rsqrt2";
    }

  return "unknown";
}

static int mode_iterations (rsqrt_kernel_mode_t mode)
{
  switch (mode)
    {
    case RSQRT_KERNEL_RSQRT1:
      return 1;
    case RSQRT_KERNEL_RSQRT2:
      return 2;
    case RSQRT_KERNEL_SQRTF:
    case RSQRT_KERNEL_RSQRT0:
      return 0;
    }

  return 0;
}

static void print_usage (const char *program)
{
  fprintf (stderr,
           "usage: %s [options]\n"
           "\n"
           "options:\n"
           "  --n N           number of positive r2 samples (default: 16777216)\n"
           "  --repeats N     repetitions per mode (default: 5)\n"
           "  --mode NAME     sqrtf, rsqrt0, rsqrt1, rsqrt2, or all (default: all)\n"
           "  --no-header     omit CSV header\n"
           "  --help          show this help message\n",
           program);
}

static void fill_inputs (float *r2, size_t n)
{
  for (size_t i = 0u; i < n; ++i)
    {
      const float t = (float) ((i * 1103515245u + 12345u) & 0xffffu)
                      / 65535.0f;
      r2[i] = 0.0025f + 64.0f * t;
    }
}

static inline float scalar_rsqrt_estimate (float x)
{
#if defined (__SSE__)
  const __m128 value = _mm_set_ss (x);
  const __m128 estimate = _mm_rsqrt_ss (value);
  return _mm_cvtss_f32 (estimate);
#else
  return 1.0f / sqrtf (x);
#endif
}

static inline float refine_scalar (float x, float y, int iterations)
{
  for (int i = 0; i < iterations; ++i)
    y = y * (1.5f - 0.5f * x * y * y);
  return y;
}

static double run_scalar_sqrtf (const float *r2, size_t n, float *checksum)
{
  double sum = 0.0;
  const double t0 = nbody_wall_seconds ();

  for (size_t i = 0u; i < n; ++i)
    sum += (double) (1.0f / sqrtf (r2[i]));

  *checksum = (float) sum;
  return nbody_wall_seconds () - t0;
}

static double run_rsqrt_mode (const float *r2,
                              size_t n,
                              rsqrt_kernel_mode_t mode,
                              float *checksum)
{
  const int iterations = mode_iterations (mode);
  double sum = 0.0;
  size_t i = 0u;
  const double t0 = nbody_wall_seconds ();

#if defined (__AVX__)
  __m256 vsum = _mm256_setzero_ps ();
  const __m256 half = _mm256_set1_ps (0.5f);
  const __m256 three_halves = _mm256_set1_ps (1.5f);

  for (; i + 8u <= n; i += 8u)
    {
      const __m256 x = _mm256_load_ps (&r2[i]);
      __m256 y = _mm256_rsqrt_ps (x);

      for (int iter = 0; iter < iterations; ++iter)
        {
          const __m256 yy = _mm256_mul_ps (y, y);
          const __m256 xyy = _mm256_mul_ps (x, yy);
          const __m256 correction =
            _mm256_sub_ps (three_halves, _mm256_mul_ps (half, xyy));
          y = _mm256_mul_ps (y, correction);
        }

      vsum = _mm256_add_ps (vsum, y);
    }

  float tmp[8] __attribute__ ((aligned (32)));
  _mm256_store_ps (tmp, vsum);
  for (int lane = 0; lane < 8; ++lane)
    sum += (double) tmp[lane];
#endif

  for (; i < n; ++i)
    {
      float y = scalar_rsqrt_estimate (r2[i]);
      y = refine_scalar (r2[i], y, iterations);
      sum += (double) y;
    }

  *checksum = (float) sum;
  return nbody_wall_seconds () - t0;
}

static float eval_mode_scalar (float x, rsqrt_kernel_mode_t mode)
{
  if (mode == RSQRT_KERNEL_SQRTF)
    return 1.0f / sqrtf (x);

  return refine_scalar (x, scalar_rsqrt_estimate (x), mode_iterations (mode));
}

static double max_relative_error (const float *r2,
                                  size_t n,
                                  rsqrt_kernel_mode_t mode)
{
  double max_error = 0.0;

  for (size_t i = 0u; i < n; ++i)
    {
      const double reference = 1.0 / sqrt ((double) r2[i]);
      const double estimate = (double) eval_mode_scalar (r2[i], mode);
      const double relative = fabs (estimate - reference) / reference;

      if (relative > max_error)
        max_error = relative;
    }

  return max_error;
}

static void run_mode (const float *r2,
                      size_t n,
                      size_t repeats,
                      rsqrt_kernel_mode_t mode)
{
  const double error = max_relative_error (r2, n, mode);

  for (size_t repeat = 1u; repeat <= repeats; ++repeat)
    {
      float checksum = 0.0f;
      double seconds;

      if (mode == RSQRT_KERNEL_SQRTF)
        seconds = run_scalar_sqrtf (r2, n, &checksum);
      else
        seconds = run_rsqrt_mode (r2, n, mode, &checksum);

      printf ("%s,%zu,%zu,%.9f,%.9g,%.17g,OK\n",
              mode_name (mode), n, repeat, seconds, checksum, error);
    }
}

int main (int argc, char **argv)
{
  size_t n = 16777216u;
  size_t repeats = 5u;
  const char *mode_text = "all";
  bool print_header = true;
  float *r2;

  for (int argi = 1; argi < argc; ++argi)
    {
      const char *value;

      if ((value = option_value (&argi, argc, argv, "--n")) != NULL)
        n = parse_size (value, "--n");
      else if ((value = option_value (&argi, argc, argv, "--repeats")) != NULL)
        repeats = parse_size (value, "--repeats");
      else if ((value = option_value (&argi, argc, argv, "--mode")) != NULL)
        mode_text = value;
      else if (strcmp (argv[argi], "--no-header") == 0)
        print_header = false;
      else if (strcmp (argv[argi], "--help") == 0)
        {
          print_usage (argv[0]);
          return EXIT_SUCCESS;
        }
      else
        nbody_die ("unknown option: %s", argv[argi]);
    }

  if (n == 0u)
    nbody_die ("--n must be positive");
  if (repeats == 0u)
    nbody_die ("--repeats must be positive");

  r2 = nbody_aligned_alloc (n * sizeof (*r2), NBODY_ALIGNMENT);
  fill_inputs (r2, n);

  if (print_header)
    printf ("method,n,repeat,seconds,checksum,max_relative_error,status\n");

  if (strcmp (mode_text, "all") == 0)
    {
      run_mode (r2, n, repeats, RSQRT_KERNEL_SQRTF);
      run_mode (r2, n, repeats, RSQRT_KERNEL_RSQRT0);
      run_mode (r2, n, repeats, RSQRT_KERNEL_RSQRT1);
      run_mode (r2, n, repeats, RSQRT_KERNEL_RSQRT2);
    }
  else
    run_mode (r2, n, repeats, parse_mode (mode_text));

  free (r2);
  return EXIT_SUCCESS;
}
