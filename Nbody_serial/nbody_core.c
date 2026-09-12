/*
 * nbody_core.c
 *
 * Shared utilities for the direct N-body programs: particle storage, binary
 * I/O, scalar parsing, timing, drift/kick updates, and energy diagnostics.
 */

#define _POSIX_C_SOURCE 200809L

#include "nbody_core.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static nbody_error_handler_t error_handler = NULL;

void nbody_set_error_handler (nbody_error_handler_t handler)
{
  error_handler = handler;
}

void nbody_die (const char *format, ...)
{
  va_list args;

  va_start (args, format);
  vfprintf (stderr, format, args);
  va_end (args);
  fputc ('\n', stderr);
  if (error_handler != NULL)
    error_handler ();
  exit (EXIT_FAILURE);
}

double nbody_wall_seconds (void)
{
  struct timespec now;

  if (clock_gettime (CLOCK_MONOTONIC, &now) != 0)
    nbody_die ("clock_gettime(CLOCK_MONOTONIC) failed");

  return (double) now.tv_sec + (double) now.tv_nsec * 1.0e-9;
}

int nbody_openmp_max_threads (void)
{
#ifdef _OPENMP
  return omp_get_max_threads ();
#else
  return 1;
#endif
}

const char *nbody_openmp_status (void)
{
#ifdef _OPENMP
  return "enabled";
#else
  return "disabled";
#endif
}

void timing_init (timing_t *timing)
{
  timing->total_seconds = 0.0;
  timing->read_seconds = 0.0;
  timing->distribute_seconds = 0.0;
  timing->initial_energy_seconds = 0.0;
  timing->initial_acceleration_seconds = 0.0;
  timing->integration_seconds = 0.0;
  timing->force_seconds = 0.0;
  timing->communication_seconds = 0.0;
  timing->drift_seconds = 0.0;
  timing->kick_seconds = 0.0;
  timing->energy_seconds = 0.0;
  timing->write_seconds = 0.0;
}

void timing_print (const timing_t *timing)
{
  printf ("# timing openmp %s max_threads %d\n",
          nbody_openmp_status (), nbody_openmp_max_threads ());
  printf ("# timing total_seconds %.9f\n", timing->total_seconds);
  printf ("# timing read_seconds %.9f\n", timing->read_seconds);
  printf ("# timing initial_energy_seconds %.9f\n", timing->initial_energy_seconds);
  printf ("# timing initial_acceleration_seconds %.9f\n", timing->initial_acceleration_seconds);
  printf ("# timing integration_seconds %.9f\n", timing->integration_seconds);
  printf ("# timing force_seconds %.9f\n", timing->force_seconds);
  printf ("# timing drift_seconds %.9f\n", timing->drift_seconds);
  printf ("# timing kick_seconds %.9f\n", timing->kick_seconds);
  printf ("# timing energy_seconds %.9f\n", timing->energy_seconds);
  printf ("# timing write_seconds %.9f\n", timing->write_seconds);
}

/*
 * Parse a size_t command-line value.  All user-facing quantities that count
 * particles or steps pass through this function so that overflow and malformed
 * input fail early, before any allocation or simulation state is modified.
 */
size_t parse_size (const char *text, const char *name)
{
  char *endptr;
  unsigned long long value;

  errno = 0;
  value = strtoull (text, &endptr, 10);
  if ((errno != 0) || (endptr == text) || (*endptr != '\0'))
    nbody_die ("invalid integer for %s: %s", name, text);
  if (value > (unsigned long long) SIZE_MAX)
    nbody_die ("integer for %s is too large: %s", name, text);

  return (size_t) value;
}

/*
 * Parse a finite floating-point command-line value and cast it to dtype.  The
 * parser reads through double because strtof and strtod differ only in final
 * rounding for the ranges used here; the explicit range check keeps float-mode
 * builds from silently accepting values that cannot be represented by dtype.
 */
dtype parse_dtype (const char *text, const char *name)
{
  char *endptr;
  double value;

  errno = 0;
  value = strtod (text, &endptr);
  if ((errno != 0) || (endptr == text) || (*endptr != '\0') || !isfinite (value))
    nbody_die ("invalid floating-point value for %s: %s", name, text);
  if (fabs (value) > (double) DTYPE_MAX_VALUE)
    nbody_die ("floating-point value for %s is outside the selected dtype range: %s", name, text);

  return (dtype) value;
}

/*
 * Return the value associated with either "--key value" or "--key=value".
 * The caller passes the loop index by address so that the separated-value
 * form consumes the following argv entry exactly once.
 */
const char *option_value (int *i, int argc, char **argv, const char *key)
{
  const size_t key_len = strlen (key);
  const char *arg = argv[*i];

  if ((strncmp (arg, key, key_len) == 0) && (arg[key_len] == '='))
    return arg + key_len + 1;

  if (strcmp (arg, key) == 0)
    {
      if (*i + 1 >= argc)
        nbody_die ("missing value after %s", key);
      *i += 1;
      return argv[*i];
    }

  return NULL;
}

void *nbody_aligned_alloc (size_t nbytes, size_t alignment)
{
  void *ptr;
  size_t padded;

  if (nbytes == 0u)
    nbody_die ("attempted zero-byte allocation");
  if (alignment == 0u)
    nbody_die ("invalid zero alignment");
  if (nbytes > SIZE_MAX - alignment)
    nbody_die ("allocation size overflow");

  padded = ((nbytes + alignment - 1u) / alignment) * alignment;
  ptr = aligned_alloc (alignment, padded);
  if (ptr == NULL)
    nbody_die ("aligned_alloc failed for %zu bytes", padded);

  return ptr;
}

static void checked_fread (void *ptr, size_t size, size_t nmemb,
                           FILE *fp, const char *path, const char *what)
{
  const size_t got = fread (ptr, size, nmemb, fp);

  if (got != nmemb)
    {
      if (ferror (fp))
        nbody_die ("read error while reading %s from '%s'", what, path);
      nbody_die ("short file while reading %s from '%s'", what, path);
    }
}

static void checked_fwrite (const void *ptr, size_t size, size_t nmemb,
                            FILE *fp, const char *path, const char *what)
{
  const size_t written = fwrite (ptr, size, nmemb, fp);

  if (written != nmemb)
    nbody_die ("write error while writing %s to '%s'", what, path);
}

void particles_init_empty (particles_t *p)
{
  p->n = 0u;
  p->mass = (dtype) 1.0;
  p->x = NULL;
  p->y = NULL;
  p->z = NULL;
  p->vx = NULL;
  p->vy = NULL;
  p->vz = NULL;
  p->ax = NULL;
  p->ay = NULL;
  p->az = NULL;
}

void particles_allocate (particles_t *p, size_t n, dtype mass)
{
  const size_t bytes = n * sizeof (dtype);

  if (n == 0u)
    nbody_die ("the number of particles must be positive");
  if (n > SIZE_MAX / sizeof (dtype))
    nbody_die ("particle count is too large");
  if (!(mass > (dtype) 0.0) || !dtype_isfinite (mass))
    nbody_die ("particle mass must be positive and finite");

  particles_init_empty (p);
  p->n = n;
  p->mass = mass;
  p->x = nbody_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->y = nbody_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->z = nbody_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vx = nbody_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vy = nbody_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vz = nbody_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->ax = nbody_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->ay = nbody_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->az = nbody_aligned_alloc (bytes, NBODY_ALIGNMENT);
}

void particles_free (particles_t *p)
{
  free (p->x);
  free (p->y);
  free (p->z);
  free (p->vx);
  free (p->vy);
  free (p->vz);
  free (p->ax);
  free (p->ay);
  free (p->az);
  particles_init_empty (p);
}

/*
 * Cast one physical quantity to the on-disk type.
 * Nofinite values and overflows should still fail loudly.
 * How do we deal with that?
 * Check the initial condition generator for a more performant
 * implementation.
 *
 * NOTE: may this be a bottleneck in the I/O ? what your profiling says?
 *       if so, you may consider to make this optional only when a "debugging mode"
 *       is set, and to expand to a simple cast otherwose,
 *       Optionally this sanity checks should be made as a loop over particles that
 *       accumulates failures counter, instead on per-particle funciton call
 *
 * Project note: keep the loud check in the default implementation for now.
 * Profiling so far shows the force/energy O(N^2) kernels dominate the runs we
 * benchmarked; if we later write many snapshots, this can become a debug-only
 * path or a batched validation loop.
 */
static float dtype_to_storage_float (dtype value, const char *component, size_t i)
{
  const double as_double = (double) value;

  if (!isfinite (as_double) || (fabs (as_double) > (double) FLT_MAX))
    nbody_die ("particle %zu component %s cannot be stored as a finite float", i, component);

  return (float) value;
}

void particles_read_binary (const char *path, dtype mass, particles_t *p)
{
  FILE *fp;
  unsigned char magic[NBODY_BINARY_MAGIC_SIZE];
  uint64_t n64;
  size_t n;

  fp = fopen (path, "rb");
  if (fp == NULL)
    nbody_die ("cannot open input file '%s'", path);

  checked_fread (magic, sizeof magic[0], NBODY_BINARY_MAGIC_SIZE,
                 fp, path, "binary magic");
  if (memcmp (magic, nbody_binary_magic, NBODY_BINARY_MAGIC_SIZE) != 0)
    nbody_die ("input file '%s' is not an %s file", path, NBODY_BINARY_VERSION_TEXT);

  checked_fread (&n64, sizeof n64, 1u, fp, path, "particle count");
  if ((n64 == 0u) || (n64 > (uint64_t) SIZE_MAX))
    nbody_die ("invalid particle count in '%s'", path);
  n = (size_t) n64;

  particles_allocate (p, n, mass);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0u; i < n; ++i)
    {
      float record[NBODY_BINARY_COMPONENTS];

      checked_fread (record, sizeof record[0], NBODY_BINARY_COMPONENTS,
                     fp, path, "particle record");

      /*
       * this check is also very costly made like that.
       * either optimize or render it optional for some debugging mode
       */
      if (!isfinite ((double) record[0]) || !isfinite ((double) record[1]) ||
          !isfinite ((double) record[2]) || !isfinite ((double) record[3]) ||
          !isfinite ((double) record[4]) || !isfinite ((double) record[5]))
        nbody_die ("non-finite particle value in '%s' at index %zu", path, i);

      p->x[i] = (dtype) record[0];
      p->y[i] = (dtype) record[1];
      p->z[i] = (dtype) record[2];
      p->vx[i] = (dtype) record[3];
      p->vy[i] = (dtype) record[4];
      p->vz[i] = (dtype) record[5];
    }

  if (fclose (fp) != 0)
    nbody_die ("error while closing input file '%s'", path);
}

void particles_write_binary (const char *path, const particles_t *p)
{
  FILE *fp;
  const size_t n = p->n;
  uint64_t n64 = (uint64_t) n;

  if ((size_t) n64 != n)
    nbody_die ("particle count cannot be represented in the binary header");

  fp = fopen (path, "wb");
  if (fp == NULL)
    nbody_die ("cannot open output file '%s'", path);

  checked_fwrite (nbody_binary_magic, sizeof nbody_binary_magic[0],
                  NBODY_BINARY_MAGIC_SIZE, fp, path, "binary magic");
  checked_fwrite (&n64, sizeof n64, 1u, fp, path, "particle count");

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0u; i < n; ++i)
    {
      float record[NBODY_BINARY_COMPONENTS];

      record[0] = dtype_to_storage_float (p->x[i], "x", i);
      record[1] = dtype_to_storage_float (p->y[i], "y", i);
      record[2] = dtype_to_storage_float (p->z[i], "z", i);
      record[3] = dtype_to_storage_float (p->vx[i], "vx", i);
      record[4] = dtype_to_storage_float (p->vy[i], "vy", i);
      record[5] = dtype_to_storage_float (p->vz[i], "vz", i);
      checked_fwrite (record, sizeof record[0], NBODY_BINARY_COMPONENTS,
                      fp, path, "particle record");
    }

  if (fclose (fp) != 0)
    nbody_die ("error while closing output file '%s'", path);
}

/*
 * Drift all particles by a time interval using the current velocities.
 * The KDK leapfrog workflow calls it once per step, after the first half-kick.
 *
 * Again: are data qualifiers missed for optimization?
 *
 * Project note: the local restrict aliases below make the non-aliasing
 * relationship explicit inside the loop, which gives the compiler more freedom
 * to vectorise the simple streaming update.
 */
void drift (particles_t *p, dtype dt)
{
  const size_t n = p->n;
  dtype * restrict x = p->x;
  dtype * restrict y = p->y;
  dtype * restrict z = p->z;
  const dtype * restrict vx = p->vx;
  const dtype * restrict vy = p->vy;
  const dtype * restrict vz = p->vz;

  for (size_t i = 0u; i < n; ++i)
    {
      x[i] += dt * vx[i];
      y[i] += dt * vy[i];
      z[i] += dt * vz[i];
    }
}

void kick (particles_t *p, dtype dt)
{
  const size_t n = p->n;
  dtype * restrict vx = p->vx;
  dtype * restrict vy = p->vy;
  dtype * restrict vz = p->vz;
  const dtype * restrict ax = p->ax;
  const dtype * restrict ay = p->ay;
  const dtype * restrict az = p->az;

  for (size_t i = 0u; i < n; ++i)
    {
      vx[i] += dt * ax[i];
      vy[i] += dt * ay[i];
      vz[i] += dt * az[i];
    }
}

dtype kinetic_energy (const particles_t *p)
{
  const size_t n = p->n;
  const dtype mass = p->mass;
  const dtype * restrict vx_arr = p->vx;
  const dtype * restrict vy_arr = p->vy;
  const dtype * restrict vz_arr = p->vz;
  long double sum = 0.0L;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:sum)
#endif
  for (size_t i = 0u; i < n; ++i)
    {
      const long double vx = (long double) vx_arr[i];
      const long double vy = (long double) vy_arr[i];
      const long double vz = (long double) vz_arr[i];

      sum += vx * vx + vy * vy + vz * vz;
    }

  return (dtype) (0.5L * (long double) mass * sum);
}

dtype potential_energy_naive (const particles_t *p, dtype g, dtype eps)
{
  const size_t n = p->n;
  const dtype eps2 = eps * eps;
  const dtype m2 = p->mass * p->mass;
  const dtype * restrict x = p->x;
  const dtype * restrict y = p->y;
  const dtype * restrict z = p->z;
  long double sum = 0.0L;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:sum)
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
          const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);

          sum -= (long double) g * (long double) m2 * (long double) invr;
        }
    }

  return (dtype) sum;
}

dtype total_energy (const particles_t *p, dtype g, dtype eps,
                    dtype *kinetic, dtype *potential)
{
  *kinetic = kinetic_energy (p);
  *potential = potential_energy_naive (p, g, eps);

  return *kinetic + *potential;
}
