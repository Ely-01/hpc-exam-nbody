/*
 * nbody_direct_serial.c
 *
 * Serial/OpenMP C11 reference implementation for the direct gravitational
 * N-body exercise.  Shared storage, I/O, timing, and energy routines live in
 * nbody_core.c so the MPI implementation can reuse them without cloning the
 * whole program.
 */

#define _POSIX_C_SOURCE 200809L

#include "nbody_core.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined (__AVX512F__)
#include <immintrin.h>
#endif

#if defined (__SSE__)
#include <xmmintrin.h>
#endif

typedef enum integrator_e
{
  INTEGRATOR_KDK,
  INTEGRATOR_DKD
} integrator_t;

typedef enum force_kernel_e
{
  FORCE_KERNEL_DIRECT,
  FORCE_KERNEL_DIRECT_SPLIT2,
  FORCE_KERNEL_DIRECT_SPLIT4,
  FORCE_KERNEL_DIRECT_SPLIT8,
  FORCE_KERNEL_DIRECT_RSQRT512_0,
  FORCE_KERNEL_DIRECT_RSQRT512_1,
  FORCE_KERNEL_DIRECT_RSQRT512_2,
  FORCE_KERNEL_NEWTON,
  FORCE_KERNEL_NEWTON_PRIVATE,
  FORCE_KERNEL_NEWTON_ATOMIC
} force_kernel_t;

typedef enum inv_sqrt_e
{
  INV_SQRT_LIBM,
  INV_SQRT_RSQRT1,
  INV_SQRT_RSQRT2,
  INV_SQRT_RSQRT3
} inv_sqrt_t;

static integrator_t parse_integrator (const char *text)
{
  if (strcmp (text, "kdk") == 0)
    return INTEGRATOR_KDK;
  if (strcmp (text, "dkd") == 0)
    return INTEGRATOR_DKD;

  nbody_die ("invalid integrator '%s': expected kdk or dkd", text);
  return INTEGRATOR_KDK;
}

static const char *integrator_name (integrator_t integrator)
{
  switch (integrator)
    {
    case INTEGRATOR_KDK:
      return "kdk";
    case INTEGRATOR_DKD:
      return "dkd";
    }

  return "unknown";
}

static force_kernel_t parse_force_kernel (const char *text)
{
  if (strcmp (text, "direct") == 0)
    return FORCE_KERNEL_DIRECT;
  if (strcmp (text, "direct-split2") == 0)
    return FORCE_KERNEL_DIRECT_SPLIT2;
  if (strcmp (text, "direct-split4") == 0)
    return FORCE_KERNEL_DIRECT_SPLIT4;
  if (strcmp (text, "direct-split8") == 0)
    return FORCE_KERNEL_DIRECT_SPLIT8;
  if (strcmp (text, "direct-rsqrt512-0") == 0)
    return FORCE_KERNEL_DIRECT_RSQRT512_0;
  if (strcmp (text, "direct-rsqrt512-1") == 0)
    return FORCE_KERNEL_DIRECT_RSQRT512_1;
  if (strcmp (text, "direct-rsqrt512-2") == 0)
    return FORCE_KERNEL_DIRECT_RSQRT512_2;
  if (strcmp (text, "newton") == 0)
    return FORCE_KERNEL_NEWTON;
  if (strcmp (text, "newton-private") == 0)
    return FORCE_KERNEL_NEWTON_PRIVATE;
  if (strcmp (text, "newton-atomic") == 0)
    return FORCE_KERNEL_NEWTON_ATOMIC;

  nbody_die ("invalid force kernel '%s': expected direct, direct-split2, direct-split4, direct-split8, direct-rsqrt512-0, direct-rsqrt512-1, direct-rsqrt512-2, newton, newton-private, or newton-atomic", text);
  return FORCE_KERNEL_DIRECT;
}

static const char *force_kernel_name (force_kernel_t kernel)
{
  switch (kernel)
    {
    case FORCE_KERNEL_DIRECT:
      return "direct";
    case FORCE_KERNEL_DIRECT_SPLIT2:
      return "direct-split2";
    case FORCE_KERNEL_DIRECT_SPLIT4:
      return "direct-split4";
    case FORCE_KERNEL_DIRECT_SPLIT8:
      return "direct-split8";
    case FORCE_KERNEL_DIRECT_RSQRT512_0:
      return "direct-rsqrt512-0";
    case FORCE_KERNEL_DIRECT_RSQRT512_1:
      return "direct-rsqrt512-1";
    case FORCE_KERNEL_DIRECT_RSQRT512_2:
      return "direct-rsqrt512-2";
    case FORCE_KERNEL_NEWTON:
      return "newton";
    case FORCE_KERNEL_NEWTON_PRIVATE:
      return "newton-private";
    case FORCE_KERNEL_NEWTON_ATOMIC:
      return "newton-atomic";
    }

  return "unknown";
}

static inv_sqrt_t parse_inv_sqrt (const char *text)
{
  if (strcmp (text, "libm") == 0)
    return INV_SQRT_LIBM;
  if (strcmp (text, "rsqrt1") == 0)
    return INV_SQRT_RSQRT1;
  if (strcmp (text, "rsqrt2") == 0)
    return INV_SQRT_RSQRT2;
  if (strcmp (text, "rsqrt3") == 0)
    return INV_SQRT_RSQRT3;

  nbody_die ("invalid inverse-sqrt mode '%s': expected libm, rsqrt1, rsqrt2, or rsqrt3", text);
  return INV_SQRT_LIBM;
}

static const char *inv_sqrt_name (inv_sqrt_t mode)
{
  switch (mode)
    {
    case INV_SQRT_LIBM:
      return "libm";
    case INV_SQRT_RSQRT1:
      return "rsqrt1";
    case INV_SQRT_RSQRT2:
      return "rsqrt2";
    case INV_SQRT_RSQRT3:
      return "rsqrt3";
    }

  return "unknown";
}

static int inv_sqrt_iterations (inv_sqrt_t mode)
{
  switch (mode)
    {
    case INV_SQRT_RSQRT1:
      return 1;
    case INV_SQRT_RSQRT2:
      return 2;
    case INV_SQRT_RSQRT3:
      return 3;
    case INV_SQRT_LIBM:
      return 0;
    }

  return 0;
}

static dtype inv_sqrt_estimate (dtype r2)
{
#if defined (__SSE__)
  const float r2f = (float) r2;

  if ((r2f > 0.0f) && isfinite ((double) r2f))
    {
      const __m128 value = _mm_set_ss (r2f);
      const __m128 estimate = _mm_rsqrt_ss (value);
      return (dtype) _mm_cvtss_f32 (estimate);
    }
#endif

  return (dtype) 1.0 / dtype_sqrt (r2);
}

static dtype inv_sqrt_value (dtype r2, inv_sqrt_t mode)
{
  dtype y;
  int iterations;

  if (mode == INV_SQRT_LIBM)
    return (dtype) 1.0 / dtype_sqrt (r2);

  y = inv_sqrt_estimate (r2);
  iterations = inv_sqrt_iterations (mode);

  for (int i = 0; i < iterations; ++i)
    y = y * ((dtype) 1.5 - (dtype) 0.5 * r2 * y * y);

  return y;
}

/*
 * Naive direct O(N^2) softened gravitational acceleration.
 *
 * This is the most interesting kernel.
 * A very transparent form: one i particle, one j loop, no Newton-third-law
 * reuse, one accumulator per component, and a scalar sqrt from libm.  That is
 * correct, but it leaves the optimisation space visible:
 *
 *   - which data qualifiers must be introduced for the input/output pointers?
 *   - exploit or deliberately avoid Newton's third law;
 *   - split the accumulators to shorten dependency chains;
 *   - use rsqrt plus Newton refinement, then quantify energy error;
 *   - block or transpose data to improve cache/TLB behaviour;
 *   - add OpenMP without atomics in the inner loop;
 *   - later replace the all-pairs loop with an MPI ring shift.
 *
 * ... reason about the needed qualifiers to unleash compiler's optimization
 *
 * Project decision: the first OpenMP version keeps the non-Newton all-pairs
 * form, so each thread owns only ax[i], ay[i], and az[i].  This avoids atomics
 * in the inner loop and keeps the kernel compatible with the later MPI
 * ring-shift decomposition.
 */
static void compute_accelerations_direct (size_t n, dtype g, dtype mass, dtype eps,
                                          inv_sqrt_t inv_sqrt_mode,
                                          const dtype * restrict x,
                                          const dtype * restrict y,
                                          const dtype * restrict z,
                                          dtype * restrict ax,
                                          dtype * restrict ay,
                                          dtype * restrict az)
{
  const dtype eps2 = eps * eps;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0u; i < n; ++i)
    {
      const dtype xi = x[i];
      const dtype yi = y[i];
      const dtype zi = z[i];
      dtype axi = (dtype) 0.0;
      dtype ayi = (dtype) 0.0;
      dtype azi = (dtype) 0.0;

      for (size_t j = 0u; j < n; ++j)
        {
          if (j != i)
            {
              const dtype dx = x[j] - xi;
              const dtype dy = y[j] - yi;
              const dtype dz = z[j] - zi;
              const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
              const dtype invr = inv_sqrt_value (r2, inv_sqrt_mode);
              const dtype s = g * mass * invr * invr * invr;

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
static inline dtype hsum512_pd (__m512d value)
{
  dtype tmp[8];
  dtype sum = (dtype) 0.0;

  _mm512_storeu_pd (tmp, value);
  for (size_t lane = 0u; lane < 8u; ++lane)
    sum += tmp[lane];

  return sum;
}
#endif

/*
 * AVX-512 direct all-pairs force kernel with double-precision rsqrt.
 *
 * This kernel is native-GENOA specific: it requires AVX-512 and double
 * precision.  It keeps the same target-ownership formulation as the production
 * direct kernel, but vectorises the inner source-particle loop and replaces the
 * scalar libm sqrt path with _mm512_rsqrt14_pd followed by 0, 1, or 2
 * Newton-Raphson refinement steps.
 */
static void compute_accelerations_direct_rsqrt512 (size_t n, dtype g,
                                                   dtype mass, dtype eps,
                                                   int refinements,
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

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0u; i < n; ++i)
    {
      const __m512d vxi = _mm512_set1_pd (x[i]);
      const __m512d vyi = _mm512_set1_pd (y[i]);
      const __m512d vzi = _mm512_set1_pd (z[i]);
      __m512d vaxi = _mm512_setzero_pd ();
      __m512d vayi = _mm512_setzero_pd ();
      __m512d vazi = _mm512_setzero_pd ();
      dtype axi = (dtype) 0.0;
      dtype ayi = (dtype) 0.0;
      dtype azi = (dtype) 0.0;
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
  nbody_die ("direct-rsqrt512 kernels require double precision and AVX-512F");
#endif
}

/*
 * Direct all-pairs variants with multiple partial accumulators.
 *
 * The baseline kernel updates the same axi/ayi/azi values for every j, creating
 * a long floating-point dependency chain.  These variants split the reduction
 * into independent lanes and combine them at the end of the i loop.  They keep
 * exactly the same mathematical all-pairs formulation; only the local reduction
 * order changes.  The split2/split4/split8 modes let us measure where the
 * throughput gain saturates.
 */
#ifdef _OPENMP
#define NBODY_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#else
#define NBODY_OMP_PARALLEL_FOR
#endif

#define DEFINE_DIRECT_SPLIT(LANES)                                                    \
static void compute_accelerations_direct_split##LANES (size_t n, dtype g, dtype mass, \
                                                       dtype eps,                    \
                                                       inv_sqrt_t inv_sqrt_mode,     \
                                                       const dtype * restrict x,      \
                                                       const dtype * restrict y,      \
                                                       const dtype * restrict z,      \
                                                       dtype * restrict ax,           \
                                                       dtype * restrict ay,           \
                                                       dtype * restrict az)           \
{                                                                                     \
  const dtype eps2 = eps * eps;                                                       \
                                                                                      \
  NBODY_OMP_PARALLEL_FOR                                                              \
  for (size_t i = 0u; i < n; ++i)                                                     \
    {                                                                                 \
      const dtype xi = x[i];                                                          \
      const dtype yi = y[i];                                                          \
      const dtype zi = z[i];                                                          \
      dtype axp[LANES];                                                               \
      dtype ayp[LANES];                                                               \
      dtype azp[LANES];                                                               \
      dtype axi = (dtype) 0.0;                                                        \
      dtype ayi = (dtype) 0.0;                                                        \
      dtype azi = (dtype) 0.0;                                                        \
      size_t j = 0u;                                                                  \
                                                                                      \
      for (size_t lane = 0u; lane < (size_t) LANES; ++lane)                           \
        {                                                                             \
          axp[lane] = (dtype) 0.0;                                                    \
          ayp[lane] = (dtype) 0.0;                                                    \
          azp[lane] = (dtype) 0.0;                                                    \
        }                                                                             \
                                                                                      \
      for (; j + (size_t) LANES <= n; j += (size_t) LANES)                            \
        {                                                                             \
          for (size_t lane = 0u; lane < (size_t) LANES; ++lane)                       \
            {                                                                         \
              const size_t k = j + lane;                                              \
                                                                                      \
              if (k != i)                                                             \
                {                                                                     \
                  const dtype dx = x[k] - xi;                                         \
                  const dtype dy = y[k] - yi;                                         \
                  const dtype dz = z[k] - zi;                                         \
                  const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;                \
                  const dtype invr = inv_sqrt_value (r2, inv_sqrt_mode);              \
                  const dtype s = g * mass * invr * invr * invr;                     \
                                                                                      \
                  axp[lane] += dx * s;                                                \
                  ayp[lane] += dy * s;                                                \
                  azp[lane] += dz * s;                                                \
                }                                                                     \
            }                                                                         \
        }                                                                             \
                                                                                      \
      for (; j < n; ++j)                                                              \
        {                                                                             \
          if (j != i)                                                                 \
            {                                                                         \
              const dtype dx = x[j] - xi;                                             \
              const dtype dy = y[j] - yi;                                             \
              const dtype dz = z[j] - zi;                                             \
              const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;                    \
              const dtype invr = inv_sqrt_value (r2, inv_sqrt_mode);                  \
              const dtype s = g * mass * invr * invr * invr;                         \
                                                                                      \
              axi += dx * s;                                                          \
              ayi += dy * s;                                                          \
              azi += dz * s;                                                          \
            }                                                                         \
        }                                                                             \
                                                                                      \
      for (size_t lane = 0u; lane < (size_t) LANES; ++lane)                           \
        {                                                                             \
          axi += axp[lane];                                                           \
          ayi += ayp[lane];                                                           \
          azi += azp[lane];                                                           \
        }                                                                             \
                                                                                      \
      ax[i] = axi;                                                                    \
      ay[i] = ayi;                                                                    \
      az[i] = azi;                                                                    \
    }                                                                                 \
}

DEFINE_DIRECT_SPLIT(2)
DEFINE_DIRECT_SPLIT(4)
DEFINE_DIRECT_SPLIT(8)

#undef DEFINE_DIRECT_SPLIT
#undef NBODY_OMP_PARALLEL_FOR

/*
 * Newton-third-law variant for controlled serial comparisons.
 *
 * Each pair is visited once and contributes equal and opposite accelerations to
 * the two particles.  This halves the number of pair evaluations, but the
 * in-place update of both i and j is not the kernel used by the OpenMP/MPI
 * implementation: a parallel version would need atomics, locks, or private
 * thread-local acceleration arrays.  We keep it as an explicit comparison
 * point for the report, not as the production MPI kernel.
 */
static void compute_accelerations_newton (size_t n, dtype g, dtype mass, dtype eps,
                                          inv_sqrt_t inv_sqrt_mode,
                                          const dtype * restrict x,
                                          const dtype * restrict y,
                                          const dtype * restrict z,
                                          dtype * restrict ax,
                                          dtype * restrict ay,
                                          dtype * restrict az)
{
  const dtype eps2 = eps * eps;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0u; i < n; ++i)
    {
      ax[i] = (dtype) 0.0;
      ay[i] = (dtype) 0.0;
      az[i] = (dtype) 0.0;
    }

  for (size_t i = 0u; i < n; ++i)
    {
      const dtype xi = x[i];
      const dtype yi = y[i];
      const dtype zi = z[i];

      for (size_t j = i + 1u; j < n; ++j)
        {
          const dtype dx = x[j] - xi;
          const dtype dy = y[j] - yi;
          const dtype dz = z[j] - zi;
          const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = inv_sqrt_value (r2, inv_sqrt_mode);
          const dtype s = g * mass * invr * invr * invr;
          const dtype axij = dx * s;
          const dtype ayij = dy * s;
          const dtype azij = dz * s;

          ax[i] += axij;
          ay[i] += ayij;
          az[i] += azij;
          ax[j] -= axij;
          ay[j] -= ayij;
          az[j] -= azij;
        }
    }
}

/*
 * OpenMP Newton-third-law experiment with private thread-local accumulators.
 *
 * The pair loop still visits only j > i, so it keeps the arithmetic saving of
 * Newton's third law.  Instead of protecting ax/ay/az with atomics, every
 * OpenMP thread writes to its own private acceleration arrays.  A final
 * reduction over the thread-private arrays produces the global acceleration.
 *
 * This removes write conflicts from the inner loop and is the intended
 * non-atomic comparison against the clean direct all-pairs kernel.  Its cost is
 * the extra memory footprint, zeroing, and reduction over n * nthreads values.
 */
static void compute_accelerations_newton_private (size_t n, dtype g, dtype mass,
                                                  dtype eps,
                                                  inv_sqrt_t inv_sqrt_mode,
                                                  const dtype * restrict x,
                                                  const dtype * restrict y,
                                                  const dtype * restrict z,
                                                  dtype * restrict ax,
                                                  dtype * restrict ay,
                                                  dtype * restrict az)
{
  const dtype eps2 = eps * eps;
  size_t nthreads = 1u;
  dtype *ax_private;
  dtype *ay_private;
  dtype *az_private;

#ifdef _OPENMP
  nthreads = (size_t) omp_get_max_threads ();
#endif

  ax_private = nbody_aligned_alloc (nthreads * n * sizeof (*ax_private),
                                    NBODY_ALIGNMENT);
  ay_private = nbody_aligned_alloc (nthreads * n * sizeof (*ay_private),
                                    NBODY_ALIGNMENT);
  az_private = nbody_aligned_alloc (nthreads * n * sizeof (*az_private),
                                    NBODY_ALIGNMENT);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t k = 0u; k < nthreads * n; ++k)
    {
      ax_private[k] = (dtype) 0.0;
      ay_private[k] = (dtype) 0.0;
      az_private[k] = (dtype) 0.0;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    size_t tid = 0u;
    dtype * restrict ax_thread;
    dtype * restrict ay_thread;
    dtype * restrict az_thread;

#ifdef _OPENMP
    tid = (size_t) omp_get_thread_num ();
#endif

    ax_thread = ax_private + tid * n;
    ay_thread = ay_private + tid * n;
    az_thread = az_private + tid * n;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (size_t i = 0u; i < n; ++i)
      {
        const dtype xi = x[i];
        const dtype yi = y[i];
        const dtype zi = z[i];

        for (size_t j = i + 1u; j < n; ++j)
          {
            const dtype dx = x[j] - xi;
            const dtype dy = y[j] - yi;
            const dtype dz = z[j] - zi;
            const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
            const dtype invr = inv_sqrt_value (r2, inv_sqrt_mode);
            const dtype s = g * mass * invr * invr * invr;
            const dtype axij = dx * s;
            const dtype ayij = dy * s;
            const dtype azij = dz * s;

            ax_thread[i] += axij;
            ay_thread[i] += ayij;
            az_thread[i] += azij;
            ax_thread[j] -= axij;
            ay_thread[j] -= ayij;
            az_thread[j] -= azij;
          }
      }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0u; i < n; ++i)
    {
      dtype axi = (dtype) 0.0;
      dtype ayi = (dtype) 0.0;
      dtype azi = (dtype) 0.0;

      for (size_t t = 0u; t < nthreads; ++t)
        {
          const size_t offset = t * n + i;

          axi += ax_private[offset];
          ayi += ay_private[offset];
          azi += az_private[offset];
        }

      ax[i] = axi;
      ay[i] = ayi;
      az[i] = azi;
    }

  free (ax_private);
  free (ay_private);
  free (az_private);
}

/*
 * OpenMP Newton-third-law experiment with atomic accumulator updates.
 *
 * This is intentionally not the production kernel.  It exposes the conflict
 * that appears when the Newton pair loop is parallelised over i: each pair
 * writes both particles, so several threads can update the same ax/ay/az entry.
 * The atomics make the race correct but quantify the synchronisation overhead.
 * This kernel intentionally violates the no-atomics production guideline and is
 * used only to measure the cost of conflict resolution.
 */
static void compute_accelerations_newton_atomic (size_t n, dtype g, dtype mass, dtype eps,
                                                 inv_sqrt_t inv_sqrt_mode,
                                                 const dtype * restrict x,
                                                 const dtype * restrict y,
                                                 const dtype * restrict z,
                                                 dtype * restrict ax,
                                                 dtype * restrict ay,
                                                 dtype * restrict az)
{
  const dtype eps2 = eps * eps;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0u; i < n; ++i)
    {
      ax[i] = (dtype) 0.0;
      ay[i] = (dtype) 0.0;
      az[i] = (dtype) 0.0;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0u; i < n; ++i)
    {
      const dtype xi = x[i];
      const dtype yi = y[i];
      const dtype zi = z[i];

      for (size_t j = i + 1u; j < n; ++j)
        {
          const dtype dx = x[j] - xi;
          const dtype dy = y[j] - yi;
          const dtype dz = z[j] - zi;
          const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = inv_sqrt_value (r2, inv_sqrt_mode);
          const dtype s = g * mass * invr * invr * invr;
          const dtype axij = dx * s;
          const dtype ayij = dy * s;
          const dtype azij = dz * s;

#ifdef _OPENMP
#pragma omp atomic update
#endif
          ax[i] += axij;
#ifdef _OPENMP
#pragma omp atomic update
#endif
          ay[i] += ayij;
#ifdef _OPENMP
#pragma omp atomic update
#endif
          az[i] += azij;
#ifdef _OPENMP
#pragma omp atomic update
#endif
          ax[j] -= axij;
#ifdef _OPENMP
#pragma omp atomic update
#endif
          ay[j] -= ayij;
#ifdef _OPENMP
#pragma omp atomic update
#endif
          az[j] -= azij;
        }
    }
}

static void compute_accelerations (particles_t *p, dtype g, dtype eps,
                                   force_kernel_t kernel,
                                   inv_sqrt_t inv_sqrt_mode)
{
  if (kernel == FORCE_KERNEL_DIRECT_SPLIT2)
    {
      compute_accelerations_direct_split2 (p->n, g, p->mass, eps, inv_sqrt_mode,
                                           p->x, p->y, p->z,
                                           p->ax, p->ay, p->az);
      return;
    }
  if (kernel == FORCE_KERNEL_DIRECT_SPLIT4)
    {
      compute_accelerations_direct_split4 (p->n, g, p->mass, eps, inv_sqrt_mode,
                                           p->x, p->y, p->z,
                                           p->ax, p->ay, p->az);
      return;
    }
  if (kernel == FORCE_KERNEL_DIRECT_SPLIT8)
    {
      compute_accelerations_direct_split8 (p->n, g, p->mass, eps, inv_sqrt_mode,
                                           p->x, p->y, p->z,
                                           p->ax, p->ay, p->az);
      return;
    }
  if (kernel == FORCE_KERNEL_DIRECT_RSQRT512_0)
    {
      compute_accelerations_direct_rsqrt512 (p->n, g, p->mass, eps, 0,
                                             p->x, p->y, p->z,
                                             p->ax, p->ay, p->az);
      return;
    }
  if (kernel == FORCE_KERNEL_DIRECT_RSQRT512_1)
    {
      compute_accelerations_direct_rsqrt512 (p->n, g, p->mass, eps, 1,
                                             p->x, p->y, p->z,
                                             p->ax, p->ay, p->az);
      return;
    }
  if (kernel == FORCE_KERNEL_DIRECT_RSQRT512_2)
    {
      compute_accelerations_direct_rsqrt512 (p->n, g, p->mass, eps, 2,
                                             p->x, p->y, p->z,
                                             p->ax, p->ay, p->az);
      return;
    }
  if (kernel == FORCE_KERNEL_NEWTON)
    {
      compute_accelerations_newton (p->n, g, p->mass, eps, inv_sqrt_mode,
                                    p->x, p->y, p->z,
                                    p->ax, p->ay, p->az);
      return;
    }
  if (kernel == FORCE_KERNEL_NEWTON_PRIVATE)
    {
      compute_accelerations_newton_private (p->n, g, p->mass, eps,
                                            inv_sqrt_mode,
                                            p->x, p->y, p->z,
                                            p->ax, p->ay, p->az);
      return;
    }
  if (kernel == FORCE_KERNEL_NEWTON_ATOMIC)
    {
      compute_accelerations_newton_atomic (p->n, g, p->mass, eps, inv_sqrt_mode,
                                           p->x, p->y, p->z,
                                           p->ax, p->ay, p->az);
      return;
    }

  compute_accelerations_direct (p->n, g, p->mass, eps, inv_sqrt_mode,
                                p->x, p->y, p->z,
                                p->ax, p->ay, p->az);
}

/*
 * Compute one KDK leapfrog step:
 *
 *   1. kick velocities by dt/2 using a(t);
 *   2. drift positions by dt using v(t + dt/2);
 *   3. compute accelerations a(t + dt);
 *   4. kick velocities by dt/2 using a(t + dt).
 *
 * The caller must compute the initial accelerations before the first step.
 * This keeps positions and velocities synchronised at integer time levels.
 */
static void leapfrog_kdk_step (particles_t *p, dtype g, dtype eps,
                               dtype dt, force_kernel_t kernel,
                               inv_sqrt_t inv_sqrt_mode,
                               timing_t *timing)
{
  double t0;

  if (timing != NULL)
    t0 = nbody_wall_seconds ();
  kick (p, (dtype) 0.5 * dt);
  if (timing != NULL)
    timing->kick_seconds += nbody_wall_seconds () - t0;

  if (timing != NULL)
    t0 = nbody_wall_seconds ();
  drift (p, dt);
  if (timing != NULL)
    timing->drift_seconds += nbody_wall_seconds () - t0;

  if (timing != NULL)
    t0 = nbody_wall_seconds ();
  compute_accelerations (p, g, eps, kernel, inv_sqrt_mode);
  if (timing != NULL)
    timing->force_seconds += nbody_wall_seconds () - t0;

  if (timing != NULL)
    t0 = nbody_wall_seconds ();
  kick (p, (dtype) 0.5 * dt);
  if (timing != NULL)
    timing->kick_seconds += nbody_wall_seconds () - t0;
}

/*
 * Compute one DKD leapfrog step:
 *
 *   1. drift positions by dt/2;
 *   2. compute accelerations at the half-step positions;
 *   3. kick velocities by dt;
 *   4. drift positions by dt/2 with the updated velocities.
 *
 * This variant is kept as a comparison point against the KDK scheme required
 * by the project specification.
 */
static void leapfrog_dkd_step (particles_t *p, dtype g, dtype eps,
                               dtype dt, force_kernel_t kernel,
                               inv_sqrt_t inv_sqrt_mode,
                               timing_t *timing)
{
  double t0;

  if (timing != NULL)
    t0 = nbody_wall_seconds ();
  drift (p, (dtype) 0.5 * dt);
  if (timing != NULL)
    timing->drift_seconds += nbody_wall_seconds () - t0;

  if (timing != NULL)
    t0 = nbody_wall_seconds ();
  compute_accelerations (p, g, eps, kernel, inv_sqrt_mode);
  if (timing != NULL)
    timing->force_seconds += nbody_wall_seconds () - t0;

  if (timing != NULL)
    t0 = nbody_wall_seconds ();
  kick (p, dt);
  if (timing != NULL)
    timing->kick_seconds += nbody_wall_seconds () - t0;

  if (timing != NULL)
    t0 = nbody_wall_seconds ();
  drift (p, (dtype) 0.5 * dt);
  if (timing != NULL)
    timing->drift_seconds += nbody_wall_seconds () - t0;
}

/*
 * Print a compact command-line reference.
 * Defaults are chosen for > small test < runs
 */
static void print_usage (const char *program)
{
  fprintf (stderr,
           "usage: %s --input FILE [options]\n"
           "\n"
           "options:\n"
           "  --input FILE              input binary particle file (%s)\n"
           "  --output FILE             optional final-state binary file\n"
           "  --nsteps N                number of integration steps (default: 10)\n"
           "  --dt X                    time step (default: 0.001)\n"
           "  --eps X                   softening length (default: 0.01)\n"
           "  --G X                     gravitational constant (default: 1)\n"
           "  --mass X                  particle mass (default: 1)\n"
           "  --integrator NAME         leapfrog variant: kdk or dkd (default: kdk)\n"
           "  --force-kernel NAME       force kernel: direct, direct-split2, direct-split4,\n"
           "                            direct-split8, direct-rsqrt512-0,\n"
           "                            direct-rsqrt512-1, direct-rsqrt512-2,\n"
           "                            newton, newton-private, or newton-atomic\n"
           "                            (default: direct)\n"
           "  --inv-sqrt NAME           inverse sqrt: libm, rsqrt1, rsqrt2, or rsqrt3 (default: libm)\n"
           "  --energy-every N          diagnostic period in steps (default: 1)\n"
           "  --energy-tol X            warning tolerance for max relative drift (default: 1e-3)\n"
           "  --timing                  print section timing summary\n"
           "  --quiet                   only print final summary\n"
           "  --help                    show this help message\n",
           program, NBODY_BINARY_VERSION_TEXT);
}

int main (int argc, char **argv)
{
  const char *input_path = NULL;
  const char *output_path = NULL;
  size_t nsteps = 10u;
  size_t energy_every = 1u;
  dtype dt = (dtype) 1.0e-3;
  dtype eps = (dtype) 1.0e-2;
  dtype g = (dtype) 1.0;
  dtype mass = (dtype) 1.0;
  dtype energy_tol = (dtype) 1.0e-3;
  bool quiet = false;
  bool timing_enabled = false;
  integrator_t integrator = INTEGRATOR_KDK;
  force_kernel_t force_kernel = FORCE_KERNEL_DIRECT;
  inv_sqrt_t inv_sqrt_mode = INV_SQRT_LIBM;
  particles_t particles;
  timing_t timing;
  dtype kinetic0;
  dtype potential0;
  dtype energy0;
  double total_start;
  double t0;

  timing_init (&timing);
  total_start = nbody_wall_seconds ();
  particles_init_empty (&particles);

  for (int argi = 1; argi < argc; ++argi)
    {
      const char *value;

      if ((value = option_value (&argi, argc, argv, "--input")) != NULL)
        input_path = value;
      else if ((value = option_value (&argi, argc, argv, "--output")) != NULL)
        output_path = value;
      else if ((value = option_value (&argi, argc, argv, "--nsteps")) != NULL)
        nsteps = parse_size (value, "--nsteps");
      else if ((value = option_value (&argi, argc, argv, "--energy-every")) != NULL)
        energy_every = parse_size (value, "--energy-every");
      else if ((value = option_value (&argi, argc, argv, "--dt")) != NULL)
        dt = parse_dtype (value, "--dt");
      else if ((value = option_value (&argi, argc, argv, "--eps")) != NULL)
        eps = parse_dtype (value, "--eps");
      else if ((value = option_value (&argi, argc, argv, "--G")) != NULL)
        g = parse_dtype (value, "--G");
      else if ((value = option_value (&argi, argc, argv, "--mass")) != NULL)
        mass = parse_dtype (value, "--mass");
      else if ((value = option_value (&argi, argc, argv, "--integrator")) != NULL)
        integrator = parse_integrator (value);
      else if ((value = option_value (&argi, argc, argv, "--force-kernel")) != NULL)
        force_kernel = parse_force_kernel (value);
      else if ((value = option_value (&argi, argc, argv, "--inv-sqrt")) != NULL)
        inv_sqrt_mode = parse_inv_sqrt (value);
      else if ((value = option_value (&argi, argc, argv, "--energy-tol")) != NULL)
        energy_tol = parse_dtype (value, "--energy-tol");
      else if (strcmp (argv[argi], "--timing") == 0)
        timing_enabled = true;
      else if (strcmp (argv[argi], "--quiet") == 0)
        quiet = true;
      else if (strcmp (argv[argi], "--help") == 0)
        {
          print_usage (argv[0]);
          return EXIT_SUCCESS;
        }
      else
        {
          print_usage (argv[0]);
          nbody_die ("unknown option: %s", argv[argi]);
        }
    }

  if (input_path == NULL)
    {
      print_usage (argv[0]);
      nbody_die ("missing required --input FILE");
    }
  if (!(dt > (dtype) 0.0))
    nbody_die ("--dt must be positive");
  if (!(eps >= (dtype) 0.0))
    nbody_die ("--eps must be non-negative");
  if (!(g > (dtype) 0.0))
    nbody_die ("--G must be positive");
  if (!(mass > (dtype) 0.0))
    nbody_die ("--mass must be positive");
  if (energy_every == 0u)
    nbody_die ("--energy-every must be positive");
  if (!(energy_tol > (dtype) 0.0))
    nbody_die ("--energy-tol must be positive");

  t0 = nbody_wall_seconds ();
  particles_read_binary (input_path, mass, &particles);
  timing.read_seconds += nbody_wall_seconds () - t0;

  t0 = nbody_wall_seconds ();
  energy0 = total_energy (&particles, g, eps, &kinetic0, &potential0);
  timing.initial_energy_seconds += nbody_wall_seconds () - t0;
  timing.energy_seconds += timing.initial_energy_seconds;

  if (!quiet)
    {
      printf ("# serial direct N-body baseline, leapfrog=%s\n",
              integrator_name (integrator));
      printf ("# arithmetic_dtype=%s binary_storage=float32 format=%s\n",
              DTYPE_NAME, NBODY_BINARY_VERSION_TEXT);
      printf ("# openmp=%s max_threads=%d\n",
              nbody_openmp_status (), nbody_openmp_max_threads ());
      printf ("# N=%zu nsteps=%zu dt=%.17g eps=%.17g G=%.17g mass=%.17g integrator=%s force_kernel=%s inv_sqrt=%s\n",
              particles.n, nsteps, (double) dt, (double) eps,
              (double) g, (double) mass, integrator_name (integrator),
              force_kernel_name (force_kernel), inv_sqrt_name (inv_sqrt_mode));
      printf ("# step time kinetic potential total rel_energy_drift\n");
      printf ("%zu %.17g %.17g %.17g %.17g %.17g\n",
              (size_t) 0u, 0.0, (double) kinetic0, (double) potential0,
              (double) energy0, 0.0);
    }

  double max_rel_drift = 0.0;

  if ((nsteps > 0u) && (integrator == INTEGRATOR_KDK))
    {
      t0 = nbody_wall_seconds ();
      compute_accelerations (&particles, g, eps, force_kernel, inv_sqrt_mode);
      timing.initial_acceleration_seconds += nbody_wall_seconds () - t0;
      timing.force_seconds += timing.initial_acceleration_seconds;
    }

  t0 = nbody_wall_seconds ();
  for (size_t step = 1u; step <= nsteps; ++step)
    {
      if (integrator == INTEGRATOR_KDK)
        leapfrog_kdk_step (&particles, g, eps, dt, force_kernel,
                           inv_sqrt_mode,
                           timing_enabled ? &timing : NULL);
      else
        leapfrog_dkd_step (&particles, g, eps, dt, force_kernel,
                           inv_sqrt_mode,
                           timing_enabled ? &timing : NULL);

      if (((step % energy_every) == 0u) || (step == nsteps))
        {
          dtype kinetic;
          dtype potential;
          const double energy_t0 = nbody_wall_seconds ();
          const dtype energy = total_energy (&particles, g, eps, &kinetic, &potential);
          const double energy_dt = nbody_wall_seconds () - energy_t0;
          const double denom = fmax (fabs ((double) energy0), (double) DTYPE_MIN_NORMAL);
          const double rel = fabs ((double) (energy - energy0)) / denom;

          timing.energy_seconds += energy_dt;

          if (rel > max_rel_drift)
            max_rel_drift = rel;
          if (!quiet)
            printf ("%zu %.17g %.17g %.17g %.17g %.17g\n",
                    step, (double) step * (double) dt, (double) kinetic,
                    (double) potential, (double) energy, rel);
        }
    }
  timing.integration_seconds += nbody_wall_seconds () - t0;

  if (output_path != NULL)
    {
      t0 = nbody_wall_seconds ();
      particles_write_binary (output_path, &particles);
      timing.write_seconds += nbody_wall_seconds () - t0;
    }

  timing.total_seconds = nbody_wall_seconds () - total_start;

  printf ("# final: N=%zu steps=%zu arithmetic_dtype=%s integrator=%s force_kernel=%s inv_sqrt=%s max_relative_energy_drift=%.17g tolerance=%.17g status=%s\n",
          particles.n, nsteps, DTYPE_NAME, integrator_name (integrator),
          force_kernel_name (force_kernel), inv_sqrt_name (inv_sqrt_mode),
          max_rel_drift, (double) energy_tol,
          (max_rel_drift <= (double) energy_tol) ? "OK" : "WARNING");

  if (max_rel_drift > (double) energy_tol)
    fprintf (stderr,
             "warning: relative energy drift %.6e exceeds tolerance %.6e; "
             "try smaller --dt, larger --eps, or better initial conditions\n",
             max_rel_drift, (double) energy_tol);

  if (timing_enabled)
    timing_print (&timing);

  particles_free (&particles);

  return EXIT_SUCCESS;
}
