#ifndef NBODY_CORE_H
#define NBODY_CORE_H

#include "nbody_common.h"

#include <stddef.h>

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
  double write_seconds;
} timing_t;

typedef void (*nbody_error_handler_t) (void);

void nbody_set_error_handler (nbody_error_handler_t handler);
void nbody_die (const char *format, ...);
double nbody_wall_seconds (void);
int nbody_openmp_max_threads (void);
const char *nbody_openmp_status (void);
void *nbody_aligned_alloc (size_t nbytes, size_t alignment);

void timing_init (timing_t *timing);
void timing_print (const timing_t *timing);

size_t parse_size (const char *text, const char *name);
dtype parse_dtype (const char *text, const char *name);
const char *option_value (int *i, int argc, char **argv, const char *key);

void particles_init_empty (particles_t *p);
void particles_allocate (particles_t *p, size_t n, dtype mass);
void particles_free (particles_t *p);
void particles_read_binary (const char *path, dtype mass, particles_t *p);
void particles_write_binary (const char *path, const particles_t *p);

void drift (particles_t *p, dtype dt);
void kick (particles_t *p, dtype dt);
dtype kinetic_energy (const particles_t *p);
dtype potential_energy_naive (const particles_t *p, dtype g, dtype eps);
dtype total_energy (const particles_t *p, dtype g, dtype eps,
                    dtype *kinetic, dtype *potential);

#endif
