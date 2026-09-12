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

typedef enum integrator_e
{
  INTEGRATOR_KDK,
  INTEGRATOR_DKD
} integrator_t;

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
              const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
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
                               dtype dt, timing_t *timing)
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
  compute_accelerations_direct (p->n, g, p->mass, eps,
                                p->x, p->y, p->z,
                                p->ax, p->ay, p->az);
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
                               dtype dt, timing_t *timing)
{
  double t0;

  if (timing != NULL)
    t0 = nbody_wall_seconds ();
  drift (p, (dtype) 0.5 * dt);
  if (timing != NULL)
    timing->drift_seconds += nbody_wall_seconds () - t0;

  if (timing != NULL)
    t0 = nbody_wall_seconds ();
  compute_accelerations_direct (p->n, g, p->mass, eps,
                                p->x, p->y, p->z,
                                p->ax, p->ay, p->az);
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
      printf ("# N=%zu nsteps=%zu dt=%.17g eps=%.17g G=%.17g mass=%.17g integrator=%s\n",
              particles.n, nsteps, (double) dt, (double) eps,
              (double) g, (double) mass, integrator_name (integrator));
      printf ("# step time kinetic potential total rel_energy_drift\n");
      printf ("%zu %.17g %.17g %.17g %.17g %.17g\n",
              (size_t) 0u, 0.0, (double) kinetic0, (double) potential0,
              (double) energy0, 0.0);
    }

  double max_rel_drift = 0.0;

  if ((nsteps > 0u) && (integrator == INTEGRATOR_KDK))
    {
      t0 = nbody_wall_seconds ();
      compute_accelerations_direct (particles.n, g, particles.mass, eps,
                                    particles.x, particles.y, particles.z,
                                    particles.ax, particles.ay, particles.az);
      timing.initial_acceleration_seconds += nbody_wall_seconds () - t0;
      timing.force_seconds += timing.initial_acceleration_seconds;
    }

  t0 = nbody_wall_seconds ();
  for (size_t step = 1u; step <= nsteps; ++step)
    {
      if (integrator == INTEGRATOR_KDK)
        leapfrog_kdk_step (&particles, g, eps, dt, timing_enabled ? &timing : NULL);
      else
        leapfrog_dkd_step (&particles, g, eps, dt, timing_enabled ? &timing : NULL);

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

  printf ("# final: N=%zu steps=%zu arithmetic_dtype=%s integrator=%s max_relative_energy_drift=%.17g tolerance=%.17g status=%s\n",
          particles.n, nsteps, DTYPE_NAME, integrator_name (integrator), max_rel_drift, (double) energy_tol,
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
