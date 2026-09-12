/*
 * nbody_mpi_omp.c
 *
 * First hybrid MPI + OpenMP direct N-body implementation.  This version keeps
 * the direct O(N^2) algorithm and uses an MPI ring-shift of source particles.
 * Each rank owns a fixed home block and accumulates accelerations only for that
 * block; OpenMP parallelizes the local home-particle loop.
 */

#define _POSIX_C_SOURCE 200809L

#include "nbody_common.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct particles_s
{
  size_t n;
  dtype mass;
  dtype *x;
  dtype *y;
  dtype *z;
  dtype *vx;
  dtype *vy;
  dtype *vz;
  dtype *ax;
  dtype *ay;
  dtype *az;
} particles_t;

typedef struct timing_s
{
  double total_seconds;
  double read_seconds;
  double distribute_seconds;
  double initial_energy_seconds;
  double initial_acceleration_seconds;
  double integration_seconds;
  double force_seconds;
  double communication_seconds;
  double drift_seconds;
  double kick_seconds;
  double energy_seconds;
} timing_t;

static MPI_Datatype dtype_mpi_type (void)
{
#if defined (NBODY_USE_FLOAT)
  return MPI_FLOAT;
#else
  return MPI_DOUBLE;
#endif
}

static void die (const char *format, ...)
{
  va_list args;

  va_start (args, format);
  vfprintf (stderr, format, args);
  va_end (args);
  fputc ('\n', stderr);
  MPI_Abort (MPI_COMM_WORLD, EXIT_FAILURE);
}

static double wall_seconds (void)
{
  struct timespec now;

  if (clock_gettime (CLOCK_MONOTONIC, &now) != 0)
    die ("clock_gettime(CLOCK_MONOTONIC) failed");

  return (double) now.tv_sec + (double) now.tv_nsec * 1.0e-9;
}

static int openmp_max_threads (void)
{
#ifdef _OPENMP
  return omp_get_max_threads ();
#else
  return 1;
#endif
}

static const char *openmp_status (void)
{
#ifdef _OPENMP
  return "enabled";
#else
  return "disabled";
#endif
}

static void timing_init (timing_t *timing)
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
}

static size_t parse_size (const char *text, const char *name)
{
  char *endptr;
  unsigned long long value;

  errno = 0;
  value = strtoull (text, &endptr, 10);
  if ((errno != 0) || (endptr == text) || (*endptr != '\0'))
    die ("invalid integer for %s: %s", name, text);
  if (value > (unsigned long long) SIZE_MAX)
    die ("integer for %s is too large: %s", name, text);

  return (size_t) value;
}

static dtype parse_dtype (const char *text, const char *name)
{
  char *endptr;
  double value;

  errno = 0;
  value = strtod (text, &endptr);
  if ((errno != 0) || (endptr == text) || (*endptr != '\0') || !isfinite (value))
    die ("invalid floating-point value for %s: %s", name, text);
  if (fabs (value) > (double) DTYPE_MAX_VALUE)
    die ("floating-point value for %s is outside the selected dtype range: %s", name, text);

  return (dtype) value;
}

static const char *option_value (int *i, int argc, char **argv, const char *key)
{
  const size_t key_len = strlen (key);
  const char *arg = argv[*i];

  if ((strncmp (arg, key, key_len) == 0) && (arg[key_len] == '='))
    return arg + key_len + 1;

  if (strcmp (arg, key) == 0)
    {
      if (*i + 1 >= argc)
        die ("missing value after %s", key);
      *i += 1;
      return argv[*i];
    }

  return NULL;
}

static void *checked_aligned_alloc (size_t nbytes, size_t alignment)
{
  void *ptr;
  size_t padded;

  if (nbytes == 0u)
    die ("attempted zero-byte allocation");
  if (nbytes > SIZE_MAX - alignment)
    die ("allocation size overflow");

  padded = ((nbytes + alignment - 1u) / alignment) * alignment;
  ptr = aligned_alloc (alignment, padded);
  if (ptr == NULL)
    die ("aligned_alloc failed for %zu bytes", padded);

  return ptr;
}

static void checked_fread (void *ptr,
                           size_t size,
                           size_t nmemb,
                           FILE *fp,
                           const char *path,
                           const char *what)
{
  const size_t got = fread (ptr, size, nmemb, fp);

  if (got != nmemb)
    {
      if (ferror (fp))
        die ("read error while reading %s from '%s'", what, path);
      die ("short file while reading %s from '%s'", what, path);
    }
}

static void particles_init_empty (particles_t *p)
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

static void particles_allocate (particles_t *p, size_t n, dtype mass)
{
  const size_t bytes = n * sizeof (dtype);

  if (n == 0u)
    die ("the number of local particles must be positive");
  if (n > SIZE_MAX / sizeof (dtype))
    die ("particle count is too large");

  particles_init_empty (p);
  p->n = n;
  p->mass = mass;
  p->x = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->y = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->z = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vx = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vy = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vz = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->ax = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->ay = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->az = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
}

static void particles_free (particles_t *p)
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

static float *read_binary_records_on_root (const char *path, uint64_t *n64_out)
{
  FILE *fp;
  unsigned char magic[NBODY_BINARY_MAGIC_SIZE];
  uint64_t n64;
  size_t nrecords;
  float *records;

  fp = fopen (path, "rb");
  if (fp == NULL)
    die ("cannot open input file '%s'", path);

  checked_fread (magic, sizeof magic[0], NBODY_BINARY_MAGIC_SIZE,
                 fp, path, "binary magic");
  if (memcmp (magic, nbody_binary_magic, NBODY_BINARY_MAGIC_SIZE) != 0)
    die ("input file '%s' is not an %s file", path, NBODY_BINARY_VERSION_TEXT);

  checked_fread (&n64, sizeof n64, 1u, fp, path, "particle count");
  if ((n64 == 0u) || (n64 > (uint64_t) SIZE_MAX))
    die ("invalid particle count in '%s'", path);
  if (n64 > (uint64_t) (SIZE_MAX / NBODY_BINARY_COMPONENTS))
    die ("particle file '%s' is too large", path);

  nrecords = (size_t) n64 * NBODY_BINARY_COMPONENTS;
  records = malloc (nrecords * sizeof records[0]);
  if (records == NULL)
    die ("allocation failed while reading '%s'", path);

  checked_fread (records, sizeof records[0], nrecords, fp, path, "particle records");

  if (fclose (fp) != 0)
    die ("error while closing input file '%s'", path);

  *n64_out = n64;
  return records;
}

static void unpack_local_records (const float *records, particles_t *p)
{
  for (size_t i = 0u; i < p->n; ++i)
    {
      const float *record = records + i * NBODY_BINARY_COMPONENTS;

      for (size_t c = 0u; c < NBODY_BINARY_COMPONENTS; ++c)
        if (!isfinite ((double) record[c]))
          die ("non-finite particle value in local record %zu", i);

      p->x[i] = (dtype) record[0];
      p->y[i] = (dtype) record[1];
      p->z[i] = (dtype) record[2];
      p->vx[i] = (dtype) record[3];
      p->vy[i] = (dtype) record[4];
      p->vz[i] = (dtype) record[5];
    }
}

static void compute_accelerations_ring (particles_t *home,
                                        dtype g,
                                        dtype eps,
                                        MPI_Comm comm,
                                        timing_t *timing)
{
  int rank;
  int nranks;
  const size_t nlocal = home->n;
  const dtype eps2 = eps * eps;
  dtype *send_x;
  dtype *send_y;
  dtype *send_z;
  dtype *recv_x;
  dtype *recv_y;
  dtype *recv_z;
  int left;
  int right;

  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &nranks);

  for (size_t i = 0u; i < nlocal; ++i)
    {
      home->ax[i] = (dtype) 0.0;
      home->ay[i] = (dtype) 0.0;
      home->az[i] = (dtype) 0.0;
    }

  send_x = checked_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  send_y = checked_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  send_z = checked_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  recv_x = checked_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  recv_y = checked_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  recv_z = checked_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);

  memcpy (send_x, home->x, nlocal * sizeof (dtype));
  memcpy (send_y, home->y, nlocal * sizeof (dtype));
  memcpy (send_z, home->z, nlocal * sizeof (dtype));

  left = (rank - 1 + nranks) % nranks;
  right = (rank + 1) % nranks;

  for (int phase = 0; phase < nranks; ++phase)
    {
      double t0 = wall_seconds ();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (size_t i = 0u; i < nlocal; ++i)
        {
          const dtype xi = home->x[i];
          const dtype yi = home->y[i];
          const dtype zi = home->z[i];
          dtype axi = home->ax[i];
          dtype ayi = home->ay[i];
          dtype azi = home->az[i];

          for (size_t j = 0u; j < nlocal; ++j)
            {
              const int source_rank = (rank - phase + nranks) % nranks;
              const size_t global_i = (size_t) rank * nlocal + i;
              const size_t global_j = (size_t) source_rank * nlocal + j;

              if (global_i != global_j)
                {
                  const dtype dx = send_x[j] - xi;
                  const dtype dy = send_y[j] - yi;
                  const dtype dz = send_z[j] - zi;
                  const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
                  const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
                  const dtype s = g * home->mass * invr * invr * invr;

                  axi += dx * s;
                  ayi += dy * s;
                  azi += dz * s;
                }
            }

          home->ax[i] = axi;
          home->ay[i] = ayi;
          home->az[i] = azi;
        }

      timing->force_seconds += wall_seconds () - t0;

      if (phase + 1 < nranks)
        {
          t0 = wall_seconds ();
          MPI_Sendrecv (send_x, (int) nlocal, dtype_mpi_type (), right, 10,
                        recv_x, (int) nlocal, dtype_mpi_type (), left, 10,
                        comm, MPI_STATUS_IGNORE);
          MPI_Sendrecv (send_y, (int) nlocal, dtype_mpi_type (), right, 11,
                        recv_y, (int) nlocal, dtype_mpi_type (), left, 11,
                        comm, MPI_STATUS_IGNORE);
          MPI_Sendrecv (send_z, (int) nlocal, dtype_mpi_type (), right, 12,
                        recv_z, (int) nlocal, dtype_mpi_type (), left, 12,
                        comm, MPI_STATUS_IGNORE);
          timing->communication_seconds += wall_seconds () - t0;

          {
            dtype *tmp;

            tmp = send_x;
            send_x = recv_x;
            recv_x = tmp;
            tmp = send_y;
            send_y = recv_y;
            recv_y = tmp;
            tmp = send_z;
            send_z = recv_z;
            recv_z = tmp;
          }
        }
    }

  free (send_x);
  free (send_y);
  free (send_z);
  free (recv_x);
  free (recv_y);
  free (recv_z);
}

static void drift (particles_t *p, dtype dt)
{
  const size_t n = p->n;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0u; i < n; ++i)
    {
      p->x[i] += dt * p->vx[i];
      p->y[i] += dt * p->vy[i];
      p->z[i] += dt * p->vz[i];
    }
}

static void kick (particles_t *p, dtype dt)
{
  const size_t n = p->n;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0u; i < n; ++i)
    {
      p->vx[i] += dt * p->ax[i];
      p->vy[i] += dt * p->ay[i];
      p->vz[i] += dt * p->az[i];
    }
}

static dtype local_kinetic_energy (const particles_t *p)
{
  long double sum = 0.0L;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:sum)
#endif
  for (size_t i = 0u; i < p->n; ++i)
    {
      const long double vx = (long double) p->vx[i];
      const long double vy = (long double) p->vy[i];
      const long double vz = (long double) p->vz[i];

      sum += vx * vx + vy * vy + vz * vz;
    }

  return (dtype) (0.5L * (long double) p->mass * sum);
}

static dtype potential_energy_ring (const particles_t *home,
                                    dtype g,
                                    dtype eps,
                                    MPI_Comm comm,
                                    timing_t *timing)
{
  int rank;
  int nranks;
  const size_t nlocal = home->n;
  const dtype eps2 = eps * eps;
  const dtype m2 = home->mass * home->mass;
  dtype *send_x;
  dtype *send_y;
  dtype *send_z;
  dtype *recv_x;
  dtype *recv_y;
  dtype *recv_z;
  int left;
  int right;
  long double local_sum = 0.0L;
  dtype local_potential;
  dtype global_potential;

  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &nranks);

  send_x = checked_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  send_y = checked_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  send_z = checked_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  recv_x = checked_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  recv_y = checked_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  recv_z = checked_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);

  memcpy (send_x, home->x, nlocal * sizeof (dtype));
  memcpy (send_y, home->y, nlocal * sizeof (dtype));
  memcpy (send_z, home->z, nlocal * sizeof (dtype));

  left = (rank - 1 + nranks) % nranks;
  right = (rank + 1) % nranks;

  for (int phase = 0; phase < nranks; ++phase)
    {
      const int source_rank = (rank - phase + nranks) % nranks;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:local_sum)
#endif
      for (size_t i = 0u; i < nlocal; ++i)
        {
          const dtype xi = home->x[i];
          const dtype yi = home->y[i];
          const dtype zi = home->z[i];
          const size_t global_i = (size_t) rank * nlocal + i;

          for (size_t j = 0u; j < nlocal; ++j)
            {
              const size_t global_j = (size_t) source_rank * nlocal + j;

              if (global_j > global_i)
                {
                  const dtype dx = send_x[j] - xi;
                  const dtype dy = send_y[j] - yi;
                  const dtype dz = send_z[j] - zi;
                  const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
                  const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);

                  local_sum -= (long double) g * (long double) m2 * (long double) invr;
                }
            }
        }

      if (phase + 1 < nranks)
        {
          double t0 = wall_seconds ();

          MPI_Sendrecv (send_x, (int) nlocal, dtype_mpi_type (), right, 20,
                        recv_x, (int) nlocal, dtype_mpi_type (), left, 20,
                        comm, MPI_STATUS_IGNORE);
          MPI_Sendrecv (send_y, (int) nlocal, dtype_mpi_type (), right, 21,
                        recv_y, (int) nlocal, dtype_mpi_type (), left, 21,
                        comm, MPI_STATUS_IGNORE);
          MPI_Sendrecv (send_z, (int) nlocal, dtype_mpi_type (), right, 22,
                        recv_z, (int) nlocal, dtype_mpi_type (), left, 22,
                        comm, MPI_STATUS_IGNORE);
          timing->communication_seconds += wall_seconds () - t0;

          {
            dtype *tmp;

            tmp = send_x;
            send_x = recv_x;
            recv_x = tmp;
            tmp = send_y;
            send_y = recv_y;
            recv_y = tmp;
            tmp = send_z;
            send_z = recv_z;
            recv_z = tmp;
          }
        }
    }

  local_potential = (dtype) local_sum;
  MPI_Allreduce (&local_potential, &global_potential, 1, dtype_mpi_type (),
                 MPI_SUM, comm);

  free (send_x);
  free (send_y);
  free (send_z);
  free (recv_x);
  free (recv_y);
  free (recv_z);

  return global_potential;
}

static dtype total_energy_mpi (const particles_t *p,
                               dtype g,
                               dtype eps,
                               MPI_Comm comm,
                               timing_t *timing)
{
  dtype local_kinetic = local_kinetic_energy (p);
  dtype global_kinetic;
  dtype global_potential = potential_energy_ring (p, g, eps, comm, timing);

  MPI_Allreduce (&local_kinetic, &global_kinetic, 1, dtype_mpi_type (), MPI_SUM, comm);

  return global_kinetic + global_potential;
}

static void print_usage (const char *program)
{
  fprintf (stderr,
           "usage: %s --input FILE [options]\n"
           "\n"
           "options:\n"
           "  --input FILE              input binary particle file (%s)\n"
           "  --nsteps N                number of KDK integration steps (default: 10)\n"
           "  --dt X                    time step (default: 0.001)\n"
           "  --eps X                   softening length (default: 0.01)\n"
           "  --G X                     gravitational constant (default: 1)\n"
           "  --mass X                  particle mass (default: 1)\n"
           "  --energy-every N          diagnostic period in steps (default: 1)\n"
           "  --energy-tol X            warning tolerance for max relative drift (default: 1e-3)\n"
           "  --timing                  print section timing summary\n"
           "  --quiet                   only rank 0 prints final summary\n"
           "  --help                    show this help message\n",
           program, NBODY_BINARY_VERSION_TEXT);
}

int main (int argc, char **argv)
{
  const char *input_path = NULL;
  size_t nsteps = 10u;
  size_t energy_every = 1u;
  dtype dt = (dtype) 1.0e-3;
  dtype eps = (dtype) 1.0e-2;
  dtype g = (dtype) 1.0;
  dtype mass = (dtype) 1.0;
  dtype energy_tol = (dtype) 1.0e-3;
  bool timing_enabled = false;
  bool quiet = false;
  int rank;
  int nranks;
  uint64_t n64 = 0u;
  size_t n = 0u;
  size_t nlocal;
  float *all_records = NULL;
  float *local_records;
  particles_t particles;
  timing_t timing;
  double total_start;
  double t0;
  dtype energy0;
  double max_rel_drift = 0.0;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &nranks);
  timing_init (&timing);
  total_start = wall_seconds ();
  particles_init_empty (&particles);

  for (int argi = 1; argi < argc; ++argi)
    {
      const char *value;

      if ((value = option_value (&argi, argc, argv, "--input")) != NULL)
        input_path = value;
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
      else if ((value = option_value (&argi, argc, argv, "--energy-tol")) != NULL)
        energy_tol = parse_dtype (value, "--energy-tol");
      else if (strcmp (argv[argi], "--timing") == 0)
        timing_enabled = true;
      else if (strcmp (argv[argi], "--quiet") == 0)
        quiet = true;
      else if (strcmp (argv[argi], "--help") == 0)
        {
          if (rank == 0)
            print_usage (argv[0]);
          MPI_Finalize ();
          return EXIT_SUCCESS;
        }
      else
        {
          print_usage (argv[0]);
          die ("unknown option: %s", argv[argi]);
        }
    }

  if (input_path == NULL)
    die ("missing required --input FILE");
  if (!(dt > (dtype) 0.0))
    die ("--dt must be positive");
  if (!(eps >= (dtype) 0.0))
    die ("--eps must be non-negative");
  if (energy_every == 0u)
    die ("--energy-every must be positive");

  if (rank == 0)
    {
      t0 = wall_seconds ();
      all_records = read_binary_records_on_root (input_path, &n64);
      timing.read_seconds += wall_seconds () - t0;
    }

  MPI_Bcast (&n64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  if ((n64 % (uint64_t) nranks) != 0u)
    die ("N=%llu must be divisible by MPI ranks=%d in this first MPI version",
         (unsigned long long) n64, nranks);

  n = (size_t) n64;
  nlocal = n / (size_t) nranks;
  if (nlocal > (size_t) (INT_MAX / NBODY_BINARY_COMPONENTS))
    die ("local block is too large for MPI_Scatter counts");

  particles_allocate (&particles, nlocal, mass);
  local_records = malloc (nlocal * NBODY_BINARY_COMPONENTS * sizeof local_records[0]);
  if (local_records == NULL)
    die ("allocation failed for local particle records");

  t0 = wall_seconds ();
  MPI_Scatter (all_records, (int) (nlocal * NBODY_BINARY_COMPONENTS), MPI_FLOAT,
               local_records, (int) (nlocal * NBODY_BINARY_COMPONENTS), MPI_FLOAT,
               0, MPI_COMM_WORLD);
  timing.distribute_seconds += wall_seconds () - t0;
  free (all_records);

  unpack_local_records (local_records, &particles);
  free (local_records);

  t0 = wall_seconds ();
  energy0 = total_energy_mpi (&particles, g, eps, MPI_COMM_WORLD, &timing);
  timing.initial_energy_seconds += wall_seconds () - t0;
  timing.energy_seconds += timing.initial_energy_seconds;

  if (!quiet && rank == 0)
    {
      printf ("# MPI direct N-body KDK baseline\n");
      printf ("# arithmetic_dtype=%s binary_storage=float32 format=%s\n",
              DTYPE_NAME, NBODY_BINARY_VERSION_TEXT);
      printf ("# mpi_ranks=%d openmp=%s max_threads=%d\n",
              nranks, openmp_status (), openmp_max_threads ());
      printf ("# N=%zu Nlocal=%zu nsteps=%zu dt=%.17g eps=%.17g G=%.17g mass=%.17g\n",
              n, nlocal, nsteps, (double) dt, (double) eps, (double) g, (double) mass);
      printf ("# step time total rel_energy_drift\n");
      printf ("%zu %.17g %.17g %.17g\n", (size_t) 0u, 0.0, (double) energy0, 0.0);
    }

  if (nsteps > 0u)
    {
      t0 = wall_seconds ();
      compute_accelerations_ring (&particles, g, eps, MPI_COMM_WORLD, &timing);
      timing.initial_acceleration_seconds += wall_seconds () - t0;
    }

  t0 = wall_seconds ();
  for (size_t step = 1u; step <= nsteps; ++step)
    {
      double section_t0;

      section_t0 = wall_seconds ();
      kick (&particles, (dtype) 0.5 * dt);
      timing.kick_seconds += wall_seconds () - section_t0;

      section_t0 = wall_seconds ();
      drift (&particles, dt);
      timing.drift_seconds += wall_seconds () - section_t0;

      compute_accelerations_ring (&particles, g, eps, MPI_COMM_WORLD, &timing);

      section_t0 = wall_seconds ();
      kick (&particles, (dtype) 0.5 * dt);
      timing.kick_seconds += wall_seconds () - section_t0;

      if (((step % energy_every) == 0u) || (step == nsteps))
        {
          dtype energy;
          double rel;
          const double denom = fmax (fabs ((double) energy0), (double) DTYPE_MIN_NORMAL);

          section_t0 = wall_seconds ();
          energy = total_energy_mpi (&particles, g, eps, MPI_COMM_WORLD, &timing);
          timing.energy_seconds += wall_seconds () - section_t0;
          rel = fabs ((double) (energy - energy0)) / denom;

          if (rel > max_rel_drift)
            max_rel_drift = rel;
          if (!quiet && rank == 0)
            printf ("%zu %.17g %.17g %.17g\n",
                    step, (double) step * (double) dt, (double) energy, rel);
        }
    }
  timing.integration_seconds += wall_seconds () - t0;

  {
    timing_t max_timing;

    MPI_Reduce (&timing, &max_timing, (int) (sizeof timing / sizeof (double)),
                MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    timing.total_seconds = wall_seconds () - total_start;
    MPI_Reduce (&timing.total_seconds, &max_timing.total_seconds, 1,
                MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
      {
        printf ("# final: N=%zu Nlocal=%zu ranks=%d steps=%zu arithmetic_dtype=%s integrator=kdk max_relative_energy_drift=%.17g tolerance=%.17g status=%s\n",
                n, nlocal, nranks, nsteps, DTYPE_NAME, max_rel_drift,
                (double) energy_tol,
                (max_rel_drift <= (double) energy_tol) ? "OK" : "WARNING");

        if (timing_enabled)
          {
            printf ("# timing mpi_ranks %d openmp %s max_threads %d\n",
                    nranks, openmp_status (), openmp_max_threads ());
            printf ("# timing total_seconds %.9f\n", max_timing.total_seconds);
            printf ("# timing read_seconds %.9f\n", max_timing.read_seconds);
            printf ("# timing distribute_seconds %.9f\n", max_timing.distribute_seconds);
            printf ("# timing initial_energy_seconds %.9f\n", max_timing.initial_energy_seconds);
            printf ("# timing initial_acceleration_seconds %.9f\n", max_timing.initial_acceleration_seconds);
            printf ("# timing integration_seconds %.9f\n", max_timing.integration_seconds);
            printf ("# timing force_seconds %.9f\n", max_timing.force_seconds);
            printf ("# timing communication_seconds %.9f\n", max_timing.communication_seconds);
            printf ("# timing drift_seconds %.9f\n", max_timing.drift_seconds);
            printf ("# timing kick_seconds %.9f\n", max_timing.kick_seconds);
            printf ("# timing energy_seconds %.9f\n", max_timing.energy_seconds);
          }
      }
  }

  particles_free (&particles);
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
