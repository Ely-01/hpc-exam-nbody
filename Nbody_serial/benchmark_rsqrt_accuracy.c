/*
 * benchmark_rsqrt_accuracy.c
 *
 * Compare AVX-512 reciprocal-square-root force kernels against the scalar
 * libm direct force kernel on the same initial particle state.
 */

#define _POSIX_C_SOURCE 200809L

#include "nbody_core.h"

#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined (__AVX512F__)
#include <immintrin.h>
#endif

static void
compute_direct_libm (size_t n, dtype g, dtype mass, dtype eps,
                     const dtype * restrict x,
                     const dtype * restrict y,
                     const dtype * restrict z,
                     dtype * restrict ax,
                     dtype * restrict ay,
                     dtype * restrict az)
{
  const dtype eps2 = eps * eps;
  const dtype gmass = g * mass;

  for (size_t i = 0u; i < n; ++i)
    {
      dtype axi = (dtype) 0.0;
      dtype ayi = (dtype) 0.0;
      dtype azi = (dtype) 0.0;

      for (size_t j = 0u; j < n; ++j)
        {
          if (j != i)
            {
              const dtype dx = x[j] - x[i];
              const dtype dy = y[j] - y[i];
              const dtype dz = z[j] - z[i];
              const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
              const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
              const dtype s = gmass * invr * invr * invr;

              axi += dx * s;
              ayi += dy * s;
              azi += dz * s;
            }
        }

      ax[i] = axi;
      ay[i] = ayi;
      az[i] = azi;
    }
}

#if defined (__AVX512F__) && !defined (NBODY_USE_FLOAT)
static inline dtype
hsum512_pd (const __m512d value)
{
  dtype tmp[8];
  dtype sum = (dtype) 0.0;

  _mm512_storeu_pd (tmp, value);
  for (size_t lane = 0u; lane < 8u; ++lane)
    sum += tmp[lane];

  return sum;
}
#endif

static void
compute_rsqrt512 (size_t n, dtype g, dtype mass, dtype eps, int refinements,
                  const dtype * restrict x,
                  const dtype * restrict y,
                  const dtype * restrict z,
                  dtype * restrict ax,
                  dtype * restrict ay,
                  dtype * restrict az)
{
#if defined (__AVX512F__) && !defined (NBODY_USE_FLOAT)
  const dtype eps2 = eps * eps;
  const dtype gmass = g * mass;
  const __m512d veps2 = _mm512_set1_pd (eps2);
  const __m512d vgmass = _mm512_set1_pd (gmass);
  const __m512d vhalf = _mm512_set1_pd ((dtype) 0.5);
  const __m512d vthree_halves = _mm512_set1_pd ((dtype) 1.5);

  for (size_t i = 0u; i < n; ++i)
    {
      const __m512d vxi = _mm512_set1_pd (x[i]);
      const __m512d vyi = _mm512_set1_pd (y[i]);
      const __m512d vzi = _mm512_set1_pd (z[i]);
      __m512d vaxi = _mm512_setzero_pd ();
      __m512d vayi = _mm512_setzero_pd ();
      __m512d vazi = _mm512_setzero_pd ();
      dtype axi;
      dtype ayi;
      dtype azi;
      size_t j = 0u;

      for (; j + 8u <= n; j += 8u)
        {
          __mmask8 active = 0xffu;
          __m512d vxj;
          __m512d vyj;
          __m512d vzj;
          __m512d vdx;
          __m512d vdy;
          __m512d vdz;
          __m512d vr2;
          __m512d vinvr;
          __m512d vs;

          if (i >= j && i < j + 8u)
            active = (__mmask8) (active & ~((__mmask8) 1u << (i - j)));

          vxj = _mm512_loadu_pd (&x[j]);
          vyj = _mm512_loadu_pd (&y[j]);
          vzj = _mm512_loadu_pd (&z[j]);
          vdx = _mm512_sub_pd (vxj, vxi);
          vdy = _mm512_sub_pd (vyj, vyi);
          vdz = _mm512_sub_pd (vzj, vzi);
          vr2 = _mm512_add_pd (
              _mm512_add_pd (_mm512_mul_pd (vdx, vdx),
                             _mm512_mul_pd (vdy, vdy)),
              _mm512_add_pd (_mm512_mul_pd (vdz, vdz), veps2));
          vinvr = _mm512_rsqrt14_pd (vr2);

          for (int iter = 0; iter < refinements; ++iter)
            {
              const __m512d yy = _mm512_mul_pd (vinvr, vinvr);
              const __m512d xyy = _mm512_mul_pd (vr2, yy);
              const __m512d correction =
                  _mm512_sub_pd (vthree_halves,
                                 _mm512_mul_pd (vhalf, xyy));
              vinvr = _mm512_mul_pd (vinvr, correction);
            }

          vs = _mm512_mul_pd (vgmass,
                              _mm512_mul_pd (vinvr,
                                             _mm512_mul_pd (vinvr, vinvr)));
          vaxi = _mm512_mask_add_pd (vaxi, active, vaxi,
                                     _mm512_mul_pd (vdx, vs));
          vayi = _mm512_mask_add_pd (vayi, active, vayi,
                                     _mm512_mul_pd (vdy, vs));
          vazi = _mm512_mask_add_pd (vazi, active, vazi,
                                     _mm512_mul_pd (vdz, vs));
        }

      axi = hsum512_pd (vaxi);
      ayi = hsum512_pd (vayi);
      azi = hsum512_pd (vazi);

      for (; j < n; ++j)
        {
          if (j != i)
            {
              const dtype dx = x[j] - x[i];
              const dtype dy = y[j] - y[i];
              const dtype dz = z[j] - z[i];
              const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
              const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
              const dtype s = gmass * invr * invr * invr;

              axi += dx * s;
              ayi += dy * s;
              azi += dz * s;
            }
        }

      ax[i] = axi;
      ay[i] = ayi;
      az[i] = azi;
    }
#else
  (void) n;
  (void) g;
  (void) mass;
  (void) eps;
  (void) refinements;
  (void) x;
  (void) y;
  (void) z;
  (void) ax;
  (void) ay;
  (void) az;
  nbody_die ("rsqrt512 accuracy benchmark requires double precision and AVX-512F");
#endif
}

static void
print_usage (const char *program)
{
  fprintf (stderr,
           "usage: %s --input FILE [options]\n"
           "\n"
           "options:\n"
           "  --input FILE    input binary particle file (%s)\n"
           "  --eps X         softening length (default: 0.05)\n"
           "  --G X           gravitational constant (default: 1)\n"
           "  --mass X        particle mass (default: 1)\n"
           "  --tiny X        denominator floor for relative errors\n"
           "  --help          show this help message\n",
           program, NBODY_BINARY_VERSION_TEXT);
}

static void
compare_and_print (const char *variant, const char *force_kernel, size_t n,
                   dtype eps, dtype mass,
                   const dtype * restrict ax_ref,
                   const dtype * restrict ay_ref,
                   const dtype * restrict az_ref,
                   const dtype * restrict ax,
                   const dtype * restrict ay,
                   const dtype * restrict az,
                   dtype tiny)
{
  long double rel_sumsq = 0.0L;
  long double ref_sumsq = 0.0L;
  dtype max_rel = (dtype) 0.0;
  dtype max_abs = (dtype) 0.0;

  for (size_t i = 0u; i < n; ++i)
    {
      const dtype dax = ax[i] - ax_ref[i];
      const dtype day = ay[i] - ay_ref[i];
      const dtype daz = az[i] - az_ref[i];
      const dtype diff_norm = dtype_sqrt (dax * dax + day * day + daz * daz);
      const dtype ref_norm = dtype_sqrt (ax_ref[i] * ax_ref[i]
                                         + ay_ref[i] * ay_ref[i]
                                         + az_ref[i] * az_ref[i]);
      const dtype denom = dtype_fmax (ref_norm, tiny);
      const dtype rel = diff_norm / denom;

      if (rel > max_rel)
        max_rel = rel;
      if (diff_norm > max_abs)
        max_abs = diff_norm;
      rel_sumsq += (long double) rel * (long double) rel;
      ref_sumsq += (long double) ref_norm * (long double) ref_norm;
    }

  printf ("%s,%s,direct,%zu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,OK\n",
          variant, force_kernel, n, (double) eps, (double) mass,
          (double) max_rel,
          (double) dtype_sqrt ((dtype) (rel_sumsq / (long double) n)),
          (double) max_abs,
          (double) dtype_sqrt ((dtype) (ref_sumsq / (long double) n)));
}

int
main (int argc, char **argv)
{
  const char *input_path = NULL;
  dtype eps = (dtype) 0.05;
  dtype g = (dtype) 1.0;
  dtype mass = (dtype) 1.0;
#ifdef NBODY_USE_FLOAT
  dtype tiny = (dtype) 1.0e-30;
#else
  dtype tiny = (dtype) 1.0e-300;
#endif
  particles_t particles;
  dtype *ax_ref = NULL;
  dtype *ay_ref = NULL;
  dtype *az_ref = NULL;

  for (int argi = 1; argi < argc; ++argi)
    {
      const char *value;
      if ((value = option_value (&argi, argc, argv, "--input")) != NULL)
        input_path = value;
      else if ((value = option_value (&argi, argc, argv, "--eps")) != NULL)
        eps = parse_dtype (value, "--eps");
      else if ((value = option_value (&argi, argc, argv, "--G")) != NULL)
        g = parse_dtype (value, "--G");
      else if ((value = option_value (&argi, argc, argv, "--mass")) != NULL)
        mass = parse_dtype (value, "--mass");
      else if ((value = option_value (&argi, argc, argv, "--tiny")) != NULL)
        tiny = parse_dtype (value, "--tiny");
      else if (strcmp (argv[argi], "--help") == 0)
        {
          print_usage (argv[0]);
          return EXIT_SUCCESS;
        }
      else
        {
          print_usage (argv[0]);
          nbody_die ("unknown option '%s'", argv[argi]);
        }
    }

  if (input_path == NULL)
    {
      print_usage (argv[0]);
      nbody_die ("--input is required");
    }

  particles_init_empty (&particles);
  particles_read_binary (input_path, mass, &particles);

  ax_ref = nbody_aligned_alloc (particles.n * sizeof (*ax_ref), 64u);
  ay_ref = nbody_aligned_alloc (particles.n * sizeof (*ay_ref), 64u);
  az_ref = nbody_aligned_alloc (particles.n * sizeof (*az_ref), 64u);

  compute_direct_libm (particles.n, g, mass, eps,
                       particles.x, particles.y, particles.z,
                       ax_ref, ay_ref, az_ref);

  printf ("variant,force_kernel,reference_kernel,n,eps,mass,"
          "max_relative_accel_error,rms_relative_accel_error,"
          "max_absolute_accel_error,reference_accel_rms,status\n");
  compare_and_print ("libm", "direct", particles.n, eps, mass,
                     ax_ref, ay_ref, az_ref,
                     ax_ref, ay_ref, az_ref, tiny);

  compute_rsqrt512 (particles.n, g, mass, eps, 0,
                    particles.x, particles.y, particles.z,
                    particles.ax, particles.ay, particles.az);
  compare_and_print ("rsqrt512-0", "direct-rsqrt512-0", particles.n, eps, mass,
                     ax_ref, ay_ref, az_ref,
                     particles.ax, particles.ay, particles.az, tiny);

  compute_rsqrt512 (particles.n, g, mass, eps, 1,
                    particles.x, particles.y, particles.z,
                    particles.ax, particles.ay, particles.az);
  compare_and_print ("rsqrt512-1", "direct-rsqrt512-1", particles.n, eps, mass,
                     ax_ref, ay_ref, az_ref,
                     particles.ax, particles.ay, particles.az, tiny);

  compute_rsqrt512 (particles.n, g, mass, eps, 2,
                    particles.x, particles.y, particles.z,
                    particles.ax, particles.ay, particles.az);
  compare_and_print ("rsqrt512-2", "direct-rsqrt512-2", particles.n, eps, mass,
                     ax_ref, ay_ref, az_ref,
                     particles.ax, particles.ay, particles.az, tiny);

  free (ax_ref);
  free (ay_ref);
  free (az_ref);
  particles_free (&particles);

  return EXIT_SUCCESS;
}
