/*
 * benchmark_layout.c
 *
 * Microbenchmark for the force-kernel memory-layout trade-off.  The production
 * solver uses SoA; this program builds an AoS copy of the same input particles
 * and times the same direct all-pairs acceleration kernel on both layouts.
 */

#define _POSIX_C_SOURCE 200809L

#include "nbody_core.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct particle_aos_s
{
  dtype x;
  dtype y;
  dtype z;
  dtype vx;
  dtype vy;
  dtype vz;
  dtype ax;
  dtype ay;
  dtype az;
} particle_aos_t;

typedef enum layout_e
{
  LAYOUT_BOTH,
  LAYOUT_SOA,
  LAYOUT_AOS
} layout_t;

static layout_t parse_layout (const char *text)
{
  if (strcmp (text, "both") == 0)
    return LAYOUT_BOTH;
  if (strcmp (text, "soa") == 0)
    return LAYOUT_SOA;
  if (strcmp (text, "aos") == 0)
    return LAYOUT_AOS;

  nbody_die ("invalid layout '%s': expected both, soa, or aos", text);
  return LAYOUT_BOTH;
}

static void print_usage (const char *program)
{
  fprintf (stderr,
           "usage: %s --input FILE [options]\n"
           "\n"
           "options:\n"
           "  --input FILE        input binary particle file (%s)\n"
           "  --layout NAME       layout to benchmark: both, soa, or aos (default: both)\n"
           "  --repeats N         repeated force evaluations per layout (default: 5)\n"
           "  --eps X             softening length (default: 0.05)\n"
           "  --G X               gravitational constant (default: 1)\n"
           "  --mass X            particle mass (default: 1)\n"
           "  --no-header         omit CSV header\n"
           "  --help              show this help message\n",
           program, NBODY_BINARY_VERSION_TEXT);
}

static particle_aos_t *aos_allocate (size_t n)
{
  if (n > SIZE_MAX / sizeof (particle_aos_t))
    nbody_die ("particle count is too large for AoS allocation");
  return nbody_aligned_alloc (n * sizeof (particle_aos_t), NBODY_ALIGNMENT);
}

static void copy_soa_to_aos (const particles_t *soa, particle_aos_t *aos)
{
  for (size_t i = 0u; i < soa->n; ++i)
    {
      aos[i].x = soa->x[i];
      aos[i].y = soa->y[i];
      aos[i].z = soa->z[i];
      aos[i].vx = soa->vx[i];
      aos[i].vy = soa->vy[i];
      aos[i].vz = soa->vz[i];
      aos[i].ax = (dtype) 0.0;
      aos[i].ay = (dtype) 0.0;
      aos[i].az = (dtype) 0.0;
    }
}

static void compute_accelerations_soa (particles_t *p, dtype g, dtype eps)
{
  const size_t n = p->n;
  const dtype mass = p->mass;
  const dtype eps2 = eps * eps;
  const dtype * restrict x = p->x;
  const dtype * restrict y = p->y;
  const dtype * restrict z = p->z;
  dtype * restrict ax = p->ax;
  dtype * restrict ay = p->ay;
  dtype * restrict az = p->az;

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

static void compute_accelerations_aos (particle_aos_t * restrict p,
                                       size_t n,
                                       dtype g,
                                       dtype mass,
                                       dtype eps)
{
  const dtype eps2 = eps * eps;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0u; i < n; ++i)
    {
      const dtype xi = p[i].x;
      const dtype yi = p[i].y;
      const dtype zi = p[i].z;
      dtype axi = (dtype) 0.0;
      dtype ayi = (dtype) 0.0;
      dtype azi = (dtype) 0.0;

      for (size_t j = 0u; j < n; ++j)
        {
          if (j != i)
            {
              const dtype dx = p[j].x - xi;
              const dtype dy = p[j].y - yi;
              const dtype dz = p[j].z - zi;
              const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
              const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
              const dtype s = g * mass * invr * invr * invr;

              axi += dx * s;
              ayi += dy * s;
              azi += dz * s;
            }
        }

      p[i].ax = axi;
      p[i].ay = ayi;
      p[i].az = azi;
    }
}

static double max_abs_accel_diff (const particles_t *soa,
                                  const particle_aos_t *aos)
{
  double max_diff = 0.0;

  for (size_t i = 0u; i < soa->n; ++i)
    {
      const double dx = fabs ((double) soa->ax[i] - (double) aos[i].ax);
      const double dy = fabs ((double) soa->ay[i] - (double) aos[i].ay);
      const double dz = fabs ((double) soa->az[i] - (double) aos[i].az);

      if (dx > max_diff)
        max_diff = dx;
      if (dy > max_diff)
        max_diff = dy;
      if (dz > max_diff)
        max_diff = dz;
    }

  return max_diff;
}

static double time_soa (particles_t *particles, dtype g, dtype eps)
{
  const double t0 = nbody_wall_seconds ();
  compute_accelerations_soa (particles, g, eps);
  return nbody_wall_seconds () - t0;
}

static double time_aos (particle_aos_t *aos, size_t n,
                        dtype g, dtype mass, dtype eps)
{
  const double t0 = nbody_wall_seconds ();
  compute_accelerations_aos (aos, n, g, mass, eps);
  return nbody_wall_seconds () - t0;
}

int main (int argc, char **argv)
{
  const char *input_path = NULL;
  size_t repeats = 5u;
  dtype eps = (dtype) 0.05;
  dtype g = (dtype) 1.0;
  dtype mass = (dtype) 1.0;
  layout_t layout = LAYOUT_BOTH;
  bool print_header = true;
  particles_t particles;
  particle_aos_t *aos = NULL;

  particles_init_empty (&particles);

  for (int argi = 1; argi < argc; ++argi)
    {
      const char *value;

      if ((value = option_value (&argi, argc, argv, "--input")) != NULL)
        input_path = value;
      else if ((value = option_value (&argi, argc, argv, "--layout")) != NULL)
        layout = parse_layout (value);
      else if ((value = option_value (&argi, argc, argv, "--repeats")) != NULL)
        repeats = parse_size (value, "--repeats");
      else if ((value = option_value (&argi, argc, argv, "--eps")) != NULL)
        eps = parse_dtype (value, "--eps");
      else if ((value = option_value (&argi, argc, argv, "--G")) != NULL)
        g = parse_dtype (value, "--G");
      else if ((value = option_value (&argi, argc, argv, "--mass")) != NULL)
        mass = parse_dtype (value, "--mass");
      else if (strcmp (argv[argi], "--no-header") == 0)
        print_header = false;
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
  if (repeats == 0u)
    nbody_die ("--repeats must be positive");
  if (!(eps >= (dtype) 0.0))
    nbody_die ("--eps must be non-negative");
  if (!(g > (dtype) 0.0))
    nbody_die ("--G must be positive");
  if (!(mass > (dtype) 0.0))
    nbody_die ("--mass must be positive");

  particles_read_binary (input_path, mass, &particles);
  aos = aos_allocate (particles.n);
  copy_soa_to_aos (&particles, aos);

  if (print_header)
    printf ("layout,n,threads,repeat,seconds,max_abs_accel_diff,status\n");

  for (size_t repeat = 1u; repeat <= repeats; ++repeat)
    {
      double soa_seconds = 0.0;
      double aos_seconds = 0.0;
      double diff;

      if ((repeat % 2u) == 1u)
        {
          if ((layout == LAYOUT_BOTH) || (layout == LAYOUT_SOA))
            soa_seconds = time_soa (&particles, g, eps);
          if ((layout == LAYOUT_BOTH) || (layout == LAYOUT_AOS))
            aos_seconds = time_aos (aos, particles.n, g, mass, eps);
        }
      else
        {
          if ((layout == LAYOUT_BOTH) || (layout == LAYOUT_AOS))
            aos_seconds = time_aos (aos, particles.n, g, mass, eps);
          if ((layout == LAYOUT_BOTH) || (layout == LAYOUT_SOA))
            soa_seconds = time_soa (&particles, g, eps);
        }

      if (layout == LAYOUT_BOTH)
        diff = max_abs_accel_diff (&particles, aos);
      else
        diff = 0.0;

      if ((layout == LAYOUT_BOTH) || (layout == LAYOUT_SOA))
        printf ("soa,%zu,%d,%zu,%.9f,%.17g,OK\n",
                particles.n, nbody_openmp_max_threads (), repeat, soa_seconds, diff);
      if ((layout == LAYOUT_BOTH) || (layout == LAYOUT_AOS))
        printf ("aos,%zu,%d,%zu,%.9f,%.17g,OK\n",
                particles.n, nbody_openmp_max_threads (), repeat, aos_seconds, diff);
    }

  free (aos);
  particles_free (&particles);

  return EXIT_SUCCESS;
}
