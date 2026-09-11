/* Reader for the N-body binary format: prints a compact
 * preview and aggregate sanity checks that are useful 
 * before and after solver changes
 */

#include "nbody_common.h"

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct stats_s
{
  long double sum_x;
  long double sum_y;
  long double sum_z;
  long double sum_vx;
  long double sum_vy;
  long double sum_vz;
  long double sum_speed;
  long double max_speed;
  long double min_x;
  long double max_x;
  long double min_y;
  long double max_y;
  long double min_z;
  long double max_z;
  uint64_t non_finite;
} stats_t;

static void die(const char *format, ...)
{
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
  exit(EXIT_FAILURE);
}

static void print_usage(const char *program)
{
  fprintf (stderr,
           "usage: %s --input FILE [options]\n"
           "\n"
           "options:\n"
           " --input FILE input binary particle file (%s)\n"
           " --head N number of particles to print (default: 5)\n"
           " --help show this help message\n",
           program, NBODY_BINARY_VERSION_TEXT);
}

static size_t parse_size (const char *text,
                          const char *name)
{
  char *endptr;
  unsigned long long value;

  errno = 0;
  value = strtoull (text, &endptr, 10);
  if ((errno != 0) || (endptr == text) || (*endptr != '\0'))
    die("invalid integer for %s: %s", name, text);
  if (value > (unsigned long long) SIZE_MAX)
    die("integer for %s is too large: %s", name, text);

  return(size_t) value;
}

static const char *option_value (int *i,
                                 int argc,
                                 char **argv,
                                 const char *key)
{
  const size_t key_len = strlen (key);
  const char *arg = argv[*i];

  if ((strncmp(arg, key, key_len) == 0) && (arg[key_len] == '='))
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

static void checked_fread (void *ptr,
                           size_t size,
                           size_t nmemb,
                           FILE *fp,
                           const char *path,
                           const char *what)
{
  const size_t  got = fread (ptr, size, nmemb, fp);
  if (got != nmemb)
    {
      if (ferror (fp))
        die ("read error while reading %s from '%s'", what, path);
      die ("short file while reading %s from '%s'", what, path);
    }
}

static void stats_init (stats_t *s)
{
  s->sum_x = 0.0L;
  s->sum_y = 0.0L;
  s->sum_z = 0.0L;
  s->sum_vx = 0.0L;
  s->sum_vy = 0.0L;
  s->sum_vz = 0.0L;
  s->sum_speed = 0.0L;
  s->max_speed = 0.0L;
  s->min_x = 0.0L;
  s->max_x = 0.0L;
  s->min_y = 0.0L;
  s->max_y = 0.0L;
  s->min_z = 0.0L;
  s->max_z = 0.0L;
  s->non_finite = 0u;
}

static void stats_update (stats_t *s,
                          const float record[NBODY_BINARY_COMPONENTS],
                          bool first)
{
  const long double x = (long double) record[0];
  const long double y = (long double) record[1];
  const long double z = (long double) record[2];
  const long double vx = (long double) record[3];
  const long double vy = (long double) record[4];
  const long double vz = (long double) record[5];
  const long double speed = sqrtl (vx * vx + vy * vy + vz * vz);

  for (size_t c = 0u; c < NBODY_BINARY_COMPONENTS; ++c)
    if (!isfinite ((double) record[c]))
      s->non_finite += 1u;

  s->sum_x += x;
  s->sum_y += y;
  s->sum_z += z;
  s->sum_vx += vx;
  s->sum_vy += vy;
  s->sum_vz += vz;
  s->sum_speed += speed;

  if (first || (x < s->min_x))
    s->min_x = x;
  if (first || (x > s->max_x))
    s->max_x = x;
  if (first || (y < s->min_y))
    s->min_y = y;
  if (first || (y > s->max_y))
    s->max_y = y;
  if (first || (z < s->min_z))
    s->min_z = z;
  if (first || (z > s->max_z))
    s->max_z = z;
  if (first || (speed > s->max_speed))
    s->max_speed = speed;
}

static void inspect_particles_binary (const char *path,
                                      size_t head)
{
  FILE *fp;
  unsigned char magic[NBODY_BINARY_MAGIC_SIZE];
  uint64_t n64;
  size_t n;
  stats_t stats;

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
  n = (size_t) n64;

  printf ("# file: %s\n", path);
  printf ("# format: %s\n", NBODY_BINARY_VERSION_TEXT);
  printf ("# particles: %" PRIu64 "\n", n64);
  printf ("# storage: float32 x y z vx vy vz\n");

  if (head > n)
    head = n;
  if (head > 0u)
    {
      printf ("# first %zu particles\n", head);
      printf ("# i x y z vx vy vz\n");
    }

  stats_init (&stats);
  for (size_t i = 0u; i < n; ++i)
    {
      float  record[NBODY_BINARY_COMPONENTS];

      checked_fread (record, sizeof record[0], NBODY_BINARY_COMPONENTS,
                     fp, path, "particle record");

      if (i < head)
        printf ("%zu %.9g %.9g %.9g %.9g %.9g %.9g\n",
                i,
                (double) record[0], (double) record[1], (double) record[2],
                (double) record[3], (double) record[4], (double) record[5]);

      stats_update (&stats, record, i == 0u);
    }

  if (fclose (fp) != 0)
    die ("error while closing input file '%s'", path);

  printf ("# stats\n");
  printf ("non_finite_values %" PRIu64 "\n", stats.non_finite);
  printf ("x_range %.17Lg %.17Lg\n", stats.min_x, stats.max_x);
  printf ("y_range %.17Lg %.17Lg\n", stats.min_y, stats.max_y);
  printf ("z_range %.17Lg %.17Lg\n", stats.min_z, stats.max_z);
  printf ("position_mean %.17Lg %.17Lg %.17Lg\n",
          stats.sum_x / (long double) n,
          stats.sum_y / (long double) n,
          stats.sum_z / (long double) n);
  printf ("velocity_mean %.17Lg %.17Lg %.17Lg\n",
          stats.sum_vx / (long double) n,
          stats.sum_vy / (long double) n,
          stats.sum_vz / (long double) n);
  printf ("speed_mean %.17Lg\n", stats.sum_speed / (long double) n);
  printf ("speed_max %.17Lg\n", stats.max_speed);
}

int main (int argc, char **argv)
{
  const char *input_path = NULL;
  size_t head = 5u;

  for (int argi = 1; argi < argc; ++argi)
    {
      const char *value;

      if ((value = option_value (&argi, argc, argv, "--input")) != NULL)
        input_path = value;
      else if ((value = option_value (&argi, argc, argv, "--head")) != NULL)
        head = parse_size (value, "--head");
      else if (strcmp (argv[argi], "--help") == 0)
        {
          print_usage (argv[0]);
          return EXIT_SUCCESS;
        }
      else
        {
          print_usage (argv[0]);
          die ("unknown option: %s", argv[argi]);
        }
    }

  if (input_path == NULL)
    {
      print_usage (argv[0]);
      die ("missing required --input FILE");
    }

  inspect_particles_binary (input_path, head);
  return EXIT_SUCCESS;
}
