/*
 * nbody_mpi_omp.c
 *
 * First hybrid MPI + OpenMP direct N-body implementation. Keeps
 * the direct O(N^2) algorithm and uses an MPI ring-shift of source particles.
 * Each rank owns a fixed home block and accumulates accelerations only for that
 * block; OpenMP parallelizes the local home-particle loop.
 */

#define _POSIX_C_SOURCE 200809L

#include "nbody_core.h"

#include <limits.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static MPI_Datatype dtype_mpi_type (void)
{
#if defined (NBODY_USE_FLOAT)
  return MPI_FLOAT;
#else
  return MPI_DOUBLE;
#endif
}

static void mpi_abort_handler (void)
{
  MPI_Abort (MPI_COMM_WORLD, EXIT_FAILURE);
}

static void mpi_die (const char *format, ...)
{
  va_list args;

  va_start (args, format);
  vfprintf (stderr, format, args);
  va_end (args);
  fputc ('\n', stderr);
  MPI_Abort (MPI_COMM_WORLD, EXIT_FAILURE);
}

static void print_timing_mpi (const timing_t *timing,
                              int nranks)
{
  printf ("# timing mpi_ranks %d openmp %s max_threads %d\n",
          nranks, nbody_openmp_status (), nbody_openmp_max_threads ());
  printf ("# timing total_seconds %.9f\n", timing->total_seconds);
  printf ("# timing read_seconds %.9f\n", timing->read_seconds);
  printf ("# timing distribute_seconds %.9f\n", timing->distribute_seconds);
  printf ("# timing initial_energy_seconds %.9f\n", timing->initial_energy_seconds);
  printf ("# timing initial_acceleration_seconds %.9f\n", timing->initial_acceleration_seconds);
  printf ("# timing integration_seconds %.9f\n", timing->integration_seconds);
  printf ("# timing force_seconds %.9f\n", timing->force_seconds);
  printf ("# timing communication_seconds %.9f\n", timing->communication_seconds);
  printf ("# timing drift_seconds %.9f\n", timing->drift_seconds);
  printf ("# timing kick_seconds %.9f\n", timing->kick_seconds);
  printf ("# timing energy_seconds %.9f\n", timing->energy_seconds);
}

static void rotate_position_block (dtype **send_x,
                                   dtype **send_y,
                                   dtype **send_z,
                                   dtype **recv_x,
                                   dtype **recv_y,
                                   dtype **recv_z,
                                   size_t nlocal,
                                   int left,
                                   int right,
                                   int tag_base,
                                   MPI_Comm comm,
                                   timing_t *timing)
{
  double t0 = nbody_wall_seconds ();

  MPI_Sendrecv (*send_x, (int) nlocal, dtype_mpi_type (), right, tag_base,
                *recv_x, (int) nlocal, dtype_mpi_type (), left, tag_base,
                comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv (*send_y, (int) nlocal, dtype_mpi_type (), right, tag_base + 1,
                *recv_y, (int) nlocal, dtype_mpi_type (), left, tag_base + 1,
                comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv (*send_z, (int) nlocal, dtype_mpi_type (), right, tag_base + 2,
                *recv_z, (int) nlocal, dtype_mpi_type (), left, tag_base + 2,
                comm, MPI_STATUS_IGNORE);
  timing->communication_seconds += nbody_wall_seconds () - t0;

  {
    dtype *tmp;

    tmp = *send_x;
    *send_x = *recv_x;
    *recv_x = tmp;
    tmp = *send_y;
    *send_y = *recv_y;
    *recv_y = tmp;
    tmp = *send_z;
    *send_z = *recv_z;
    *recv_z = tmp;
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

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0u; i < nlocal; ++i)
    {
      home->ax[i] = (dtype) 0.0;
      home->ay[i] = (dtype) 0.0;
      home->az[i] = (dtype) 0.0;
    }

  send_x = nbody_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  send_y = nbody_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  send_z = nbody_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  recv_x = nbody_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  recv_y = nbody_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  recv_z = nbody_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);

  memcpy (send_x, home->x, nlocal * sizeof (dtype));
  memcpy (send_y, home->y, nlocal * sizeof (dtype));
  memcpy (send_z, home->z, nlocal * sizeof (dtype));

  left = (rank - 1 + nranks) % nranks;
  right = (rank + 1) % nranks;

  for (int phase = 0; phase < nranks; ++phase)
    {
      const int source_rank = (rank - phase + nranks) % nranks;
      double t0 = nbody_wall_seconds ();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (size_t i = 0u; i < nlocal; ++i)
        {
          const dtype xi = home->x[i];
          const dtype yi = home->y[i];
          const dtype zi = home->z[i];
          const size_t global_i = (size_t) rank * nlocal + i;
          dtype axi = home->ax[i];
          dtype ayi = home->ay[i];
          dtype azi = home->az[i];

          for (size_t j = 0u; j < nlocal; ++j)
            {
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

      timing->force_seconds += nbody_wall_seconds () - t0;

      if (phase + 1 < nranks)
        rotate_position_block (&send_x, &send_y, &send_z,
                               &recv_x, &recv_y, &recv_z,
                               nlocal, left, right, 10, comm, timing);
    }

  free (send_x);
  free (send_y);
  free (send_z);
  free (recv_x);
  free (recv_y);
  free (recv_z);
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

  send_x = nbody_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  send_y = nbody_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  send_z = nbody_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  recv_x = nbody_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  recv_y = nbody_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);
  recv_z = nbody_aligned_alloc (nlocal * sizeof (dtype), NBODY_ALIGNMENT);

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
        rotate_position_block (&send_x, &send_y, &send_z,
                               &recv_x, &recv_y, &recv_z,
                               nlocal, left, right, 20, comm, timing);
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
  dtype local_kinetic = kinetic_energy (p);
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
  particles_t root_particles;
  particles_t particles;
  timing_t timing;
  double total_start;
  double t0;
  dtype energy0;
  double max_rel_drift = 0.0;

  MPI_Init (&argc, &argv);
  nbody_set_error_handler (mpi_abort_handler);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &nranks);

  timing_init (&timing);
  total_start = nbody_wall_seconds ();
  particles_init_empty (&root_particles);
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
          particles_free (&root_particles);
          particles_free (&particles);
          MPI_Finalize ();
          return EXIT_SUCCESS;
        }
      else
        {
          if (rank == 0)
            print_usage (argv[0]);
          mpi_die ("unknown option: %s", argv[argi]);
        }
    }

  if (input_path == NULL)
    {
      if (rank == 0)
        print_usage (argv[0]);
      mpi_die ("missing required --input FILE");
    }
  if (!(dt > (dtype) 0.0))
    mpi_die ("--dt must be positive");
  if (!(eps >= (dtype) 0.0))
    mpi_die ("--eps must be non-negative");
  if (!(g > (dtype) 0.0))
    mpi_die ("--G must be positive");
  if (!(mass > (dtype) 0.0))
    mpi_die ("--mass must be positive");
  if (energy_every == 0u)
    mpi_die ("--energy-every must be positive");
  if (!(energy_tol > (dtype) 0.0))
    mpi_die ("--energy-tol must be positive");

  if (rank == 0)
    {
      t0 = nbody_wall_seconds ();
      particles_read_binary (input_path, mass, &root_particles);
      timing.read_seconds += nbody_wall_seconds () - t0;
      n64 = (uint64_t) root_particles.n;
    }

  MPI_Bcast (&n64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  if ((n64 % (uint64_t) nranks) != 0u)
    mpi_die ("N=%llu must be divisible by MPI ranks=%d in this first MPI version",
             (unsigned long long) n64, nranks);

  n = (size_t) n64;
  nlocal = n / (size_t) nranks;
  if (nlocal > (size_t) INT_MAX)
    mpi_die ("local block is too large for MPI scatter counts");

  particles_allocate (&particles, nlocal, mass);

  t0 = nbody_wall_seconds ();
  MPI_Scatter (rank == 0 ? root_particles.x : NULL, (int) nlocal, dtype_mpi_type (),
               particles.x, (int) nlocal, dtype_mpi_type (), 0, MPI_COMM_WORLD);
  MPI_Scatter (rank == 0 ? root_particles.y : NULL, (int) nlocal, dtype_mpi_type (),
               particles.y, (int) nlocal, dtype_mpi_type (), 0, MPI_COMM_WORLD);
  MPI_Scatter (rank == 0 ? root_particles.z : NULL, (int) nlocal, dtype_mpi_type (),
               particles.z, (int) nlocal, dtype_mpi_type (), 0, MPI_COMM_WORLD);
  MPI_Scatter (rank == 0 ? root_particles.vx : NULL, (int) nlocal, dtype_mpi_type (),
               particles.vx, (int) nlocal, dtype_mpi_type (), 0, MPI_COMM_WORLD);
  MPI_Scatter (rank == 0 ? root_particles.vy : NULL, (int) nlocal, dtype_mpi_type (),
               particles.vy, (int) nlocal, dtype_mpi_type (), 0, MPI_COMM_WORLD);
  MPI_Scatter (rank == 0 ? root_particles.vz : NULL, (int) nlocal, dtype_mpi_type (),
               particles.vz, (int) nlocal, dtype_mpi_type (), 0, MPI_COMM_WORLD);
  timing.distribute_seconds += nbody_wall_seconds () - t0;

  if (rank == 0)
    particles_free (&root_particles);

  t0 = nbody_wall_seconds ();
  energy0 = total_energy_mpi (&particles, g, eps, MPI_COMM_WORLD, &timing);
  timing.initial_energy_seconds += nbody_wall_seconds () - t0;
  timing.energy_seconds += timing.initial_energy_seconds;

  if (!quiet && rank == 0)
    {
      printf ("# MPI direct N-body KDK baseline\n");
      printf ("# arithmetic_dtype=%s binary_storage=float32 format=%s\n",
              DTYPE_NAME, NBODY_BINARY_VERSION_TEXT);
      printf ("# mpi_ranks=%d openmp=%s max_threads=%d\n",
              nranks, nbody_openmp_status (), nbody_openmp_max_threads ());
      printf ("# N=%zu Nlocal=%zu nsteps=%zu dt=%.17g eps=%.17g G=%.17g mass=%.17g\n",
              n, nlocal, nsteps, (double) dt, (double) eps, (double) g, (double) mass);
      printf ("# step time total rel_energy_drift\n");
      printf ("%zu %.17g %.17g %.17g\n", (size_t) 0u, 0.0, (double) energy0, 0.0);
    }

  if (nsteps > 0u)
    {
      t0 = nbody_wall_seconds ();
      compute_accelerations_ring (&particles, g, eps, MPI_COMM_WORLD, &timing);
      timing.initial_acceleration_seconds += nbody_wall_seconds () - t0;
    }

  t0 = nbody_wall_seconds ();
  for (size_t step = 1u; step <= nsteps; ++step)
    {
      double section_t0;

      section_t0 = nbody_wall_seconds ();
      kick (&particles, (dtype) 0.5 * dt);
      timing.kick_seconds += nbody_wall_seconds () - section_t0;

      section_t0 = nbody_wall_seconds ();
      drift (&particles, dt);
      timing.drift_seconds += nbody_wall_seconds () - section_t0;

      compute_accelerations_ring (&particles, g, eps, MPI_COMM_WORLD, &timing);

      section_t0 = nbody_wall_seconds ();
      kick (&particles, (dtype) 0.5 * dt);
      timing.kick_seconds += nbody_wall_seconds () - section_t0;

      if (((step % energy_every) == 0u) || (step == nsteps))
        {
          dtype energy;
          double rel;
          const double denom = fmax (fabs ((double) energy0), (double) DTYPE_MIN_NORMAL);

          section_t0 = nbody_wall_seconds ();
          energy = total_energy_mpi (&particles, g, eps, MPI_COMM_WORLD, &timing);
          timing.energy_seconds += nbody_wall_seconds () - section_t0;
          rel = fabs ((double) (energy - energy0)) / denom;

          if (rel > max_rel_drift)
            max_rel_drift = rel;
          if (!quiet && rank == 0)
            printf ("%zu %.17g %.17g %.17g\n",
                    step, (double) step * (double) dt, (double) energy, rel);
        }
    }
  timing.integration_seconds += nbody_wall_seconds () - t0;

  {
    timing_t max_timing;

    MPI_Reduce (&timing, &max_timing, (int) (sizeof timing / sizeof (double)),
                MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    timing.total_seconds = nbody_wall_seconds () - total_start;
    MPI_Reduce (&timing.total_seconds, &max_timing.total_seconds, 1,
                MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
      {
        printf ("# final: N=%zu Nlocal=%zu ranks=%d steps=%zu arithmetic_dtype=%s integrator=kdk max_relative_energy_drift=%.17g tolerance=%.17g status=%s\n",
                n, nlocal, nranks, nsteps, DTYPE_NAME, max_rel_drift,
                (double) energy_tol,
                (max_rel_drift <= (double) energy_tol) ? "OK" : "WARNING");

        if (timing_enabled)
          print_timing_mpi (&max_timing, nranks);
      }
  }

  particles_free (&root_particles);
  particles_free (&particles);
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
