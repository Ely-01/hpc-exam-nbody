# Serial C11 direct N-body baseline

This directory contains three stand-alone programs for the direct gravitational N-body exercise:

- `nbody_direct_serial.c`: serial softened direct solver with selectable KDK
  or DKD leapfrog integration and selectable direct/Newton/Newton-atomic force
  kernels.
- `benchmark_layout.c`: AoS-vs-SoA microbenchmark for the direct force kernel.
  step and a relative energy-drift verifier.
- `gen_plummer_sphere.c`: Plummer-sphere initial-condition generator.
- `gen_uniform_ball_maxwell.c`: uniform-ball generator with isotropic Maxwellian
  velocities.

The codes are intended as *almost complete* exam skeletons. The default direct force kernel is deliberately correct but naive. It uses an O(N^2) all-pairs loop, scalar `sqrt`, one accumulator per component, and no Newton-third-law reuse. 
The comments in `compute_accelerations_direct` mark this as the kernel whose optimization is part of the assignment, along with the hybrid parallelization. A serial Newton-third-law kernel is available with `--force-kernel newton` as a controlled comparison point; `--force-kernel newton-atomic` parallelizes the Newton pair loop with atomic accumulator updates to measure write-conflict overhead. Neither Newton variant is the MPI production kernel because pair reuse introduces non-local acceleration updates.

## Arithmetic type

All physical quantities in the solver and generators use the typedef `dtype`, defined in `nbody_common.h`.

Default build, double-precision arithmetic:

```sh
make
```

Single-precision arithmetic:

```sh
make clean
make PRECISION=float
```

OpenMP force-kernel build:

```sh
make clean
make OPENMP=1
OMP_NUM_THREADS=4 ./nbody_direct_serial --input plummer_1000.bin --nsteps 100 --dt 1e-4 --eps 0.05 --energy-every 10 --timing
```

Compare the default all-pairs force kernel with the serial Newton-third-law
variant:

```sh
./nbody_direct_serial --input plummer_4096.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --force-kernel direct --timing --quiet
./nbody_direct_serial --input plummer_4096.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --force-kernel newton --timing --quiet
./nbody_direct_serial --input plummer_4096.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --force-kernel newton-atomic --timing --quiet
```

Run the repeated Newton trade-off benchmark and generate a summary plus SVG:

```sh
THREADS="1 2 4 8" REPEATS=5 N=4096 NSTEPS=10 scripts/benchmark_newton_tradeoff.sh
python3 scripts/analyze_newton_tradeoff.py results/newton_tradeoff_YYYYMMDD_HHMMSS.csv \
  --csv results/newton_tradeoff_summary.csv \
  --markdown results/newton_tradeoff_summary.md \
  --svg report/figures/newton_tradeoff.svg
```

Compare the reference `1/sqrt` path against approximate reciprocal-sqrt modes:

```sh
./nbody_direct_serial --input plummer_4096.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --inv-sqrt libm --timing --quiet
./nbody_direct_serial --input plummer_4096.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --inv-sqrt rsqrt2 --timing --quiet
```

Run the repeated reciprocal-sqrt benchmark and generate a summary plus SVG:

```sh
THREADS="1 2 4 8" REPEATS=5 N=4096 NSTEPS=10 scripts/benchmark_rsqrt_tradeoff.sh
python3 scripts/analyze_rsqrt_tradeoff.py results/rsqrt_tradeoff_YYYYMMDD_HHMMSS.csv \
  --csv results/rsqrt_tradeoff_summary.csv \
  --markdown results/rsqrt_tradeoff_summary.md \
  --svg report/figures/rsqrt_tradeoff.svg
```

Run the AoS-vs-SoA force-kernel layout microbenchmark:

```sh
make OPENMP=1 PRECISION=double benchmark_layout
./generate_ic --model 0 --n 4096 --seed 123 --scale 1.0 --mass 1.0 --output results/plummer_layout_4096.bin
{
  OMP_NUM_THREADS=1 ./benchmark_layout --input results/plummer_layout_4096.bin --repeats 5
  OMP_NUM_THREADS=2 ./benchmark_layout --input results/plummer_layout_4096.bin --repeats 5 --no-header
  OMP_NUM_THREADS=4 ./benchmark_layout --input results/plummer_layout_4096.bin --repeats 5 --no-header
  OMP_NUM_THREADS=8 ./benchmark_layout --input results/plummer_layout_4096.bin --repeats 5 --no-header
} > results/layout_tradeoff.csv
python3 scripts/analyze_layout_tradeoff.py results/layout_tradeoff.csv \
  --csv results/layout_tradeoff_summary.csv \
  --markdown results/layout_tradeoff_summary.md \
  --svg report/figures/layout_tradeoff.svg
```

Run the accumulator-splitting force-kernel benchmark:

```sh
REPEATS=5 THREADS="1 2 4 8" N=4096 NSTEPS=10 sh scripts/benchmark_accumulator_tradeoff.sh
python3 scripts/analyze_accumulator_tradeoff.py results/accumulator_tradeoff_YYYYMMDD_HHMMSS.csv \
  --csv results/accumulator_tradeoff_summary.csv \
  --markdown results/accumulator_tradeoff_summary.md \
  --svg report/figures/accumulator_tradeoff.svg
```

For the final Orfeo run, prefer node-specific optimization flags so that the
compiler can use the instruction set of the allocated CPU:

```sh
CFLAGS="-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic" \
  REPEATS=5 THREADS="1 2 4 8" N=8192 NSTEPS=20 \
  sh scripts/benchmark_accumulator_tradeoff.sh
```

The `direct-split2`, `direct-split4`, and `direct-split8` kernels keep the same
direct all-pairs force law but split the local `ax/ay/az` accumulation into
multiple independent partial sums. This is the benchmark used to discuss the
FMA-throughput/critical-path point from the assignment: fewer dependencies can
increase throughput, but the gain eventually saturates because of extra loop
bookkeeping, register pressure, and the remaining non-accumulation work.

Run a repeated OpenMP benchmark and save a CSV under `results/`:

```sh
REPEATS=5 THREADS="1 2 4 8" N=4096 NSTEPS=10 scripts/benchmark_openmp.sh
```

Build the first MPI + OpenMP ring-shift solver when an MPI compiler wrapper is
available:

```sh
make mpi OPENMP=1
mpirun -np 4 ./nbody_mpi_omp --input plummer_8192.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --timing
```

The MPI ring communication mode is selectable:

```sh
mpirun -np 4 ./nbody_mpi_omp --input plummer_8192.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --ring-mode blocking --timing --quiet
mpirun -np 4 ./nbody_mpi_omp --input plummer_8192.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --ring-mode overlap --timing --quiet
```

`blocking` uses the original `MPI_Sendrecv` ring shift. `overlap` posts
`MPI_Irecv`/`MPI_Isend` for the next ring block before computing the current
block, then waits only before swapping buffers. The force calculation is shared
between the two modes, so the comparison isolates the communication strategy.

On SLURM systems such as Leonardo or Orfeo, the launch command will usually be
`srun` inside a batch script after loading the appropriate compiler and MPI
modules.

The repository also includes a conservative MPI smoke test for the first
cluster validation:

```sh
scripts/run_mpi_smoke.sh
sbatch scripts/slurm_mpi_smoke.slurm
```

After the smoke test, compare the serial KDK solver with MPI KDK runs on the
same initial conditions:

```sh
scripts/compare_serial_mpi.sh
sbatch scripts/slurm_compare_serial_mpi.slurm
```

Run a first hybrid MPI + OpenMP benchmark over rank/thread combinations:

```sh
CONFIGS="1x4 2x2 4x1" REPEATS=5 N=4096 NSTEPS=10 sh scripts/benchmark_hybrid.sh
sbatch scripts/slurm_hybrid_benchmark.slurm
python3 scripts/summarize_hybrid_csv.py results/hybrid_benchmark_YYYYMMDD_HHMMSS.csv
```

To compare blocking versus overlap, run the same benchmark twice with a single
ring mode per CSV:

```sh
RING_MODE=blocking CONFIGS="4x1 4x2 8x1" REPEATS=5 N=8192 NSTEPS=20 sh scripts/benchmark_hybrid.sh
RING_MODE=overlap CONFIGS="4x1 4x2 8x1" REPEATS=5 N=8192 NSTEPS=20 sh scripts/benchmark_hybrid.sh
python3 scripts/analyze_ring_overlap.py \
  results/hybrid_benchmark_blocking_YYYYMMDD_HHMMSS.csv \
  results/hybrid_benchmark_overlap_YYYYMMDD_HHMMSS.csv \
  --csv results/ring_overlap_summary.csv \
  --markdown results/ring_overlap_summary.md \
  --svg report/figures/ring_overlap.svg
```

Run strong or weak scaling benchmarks:

```sh
MODE=strong N=32768 NSTEPS=20 REPEATS=5 CONFIGS="1x1 2x1 4x1 8x1" sh scripts/benchmark_scaling.sh
MODE=weak NLOCAL=4096 NSTEPS=20 REPEATS=5 CONFIGS="1x1 2x1 4x1 8x1" sh scripts/benchmark_scaling.sh
sbatch scripts/slurm_scaling_benchmark.slurm
python3 scripts/summarize_scaling_csv.py results/strong_scaling_YYYYMMDD_HHMMSS.csv --markdown results/strong_scaling_summary.md --csv results/strong_scaling_summary.csv
python3 scripts/plot_scaling_svg.py results/strong_scaling_summary.csv --output results/strong_scaling.svg
```

Override the defaults with environment variables, for example:

```sh
MPI_RANKS=4 OMP_THREADS=2 N=128 NSTEPS=5 sh scripts/run_mpi_smoke.sh
```

The equivalent manual switches are:

```sh
-DNBODY_USE_DOUBLE
-DNBODY_USE_FLOAT
```

Only one of the two should be defined. If neither is defined, the header falls back to double precision.

## Binary file format

All programs use the same native-endian binary format. Particle data are stored in single precision, independently of the selected `dtype` used for arithmetic:

```text
byte 0..7       magic: "NBODYF1\0"
next 8 bytes    uint64_t particle count N
then N records  x y z vx vy vz, six float values per particle
```

The solver assigns one mass to every particle through `--mass`; mass is not stored per particle in the file. This keeps the initial-condition file compact and makes the equal-mass assumption explicit in the command line.

Because the format is deliberately minimal and native-endian, it is intended for same-machine teaching runs and benchmarks, not for long-term archival exchange between heterogeneous systems.

## Build

```sh
make
```

or explicitly:

```sh
cc -std=c11 -DNBODY_USE_DOUBLE -O2 -Wall -Wextra -Wpedantic nbody_direct_serial.c -lm -o nbody_direct_serial
cc -std=c11 -DNBODY_USE_DOUBLE -O2 -Wall -Wextra -Wpedantic gen_plummer_sphere.c -lm -o gen_plummer_sphere
cc -std=c11 -DNBODY_USE_DOUBLE -O2 -Wall -Wextra -Wpedantic gen_uniform_ball_maxwell.c -lm -o gen_uniform_ball_maxwell
```


Part of the assignment is to determine the best compiler’s flags and options, and the CPU bindings. List them in the final report.

## Example runs

Generate a small Plummer sphere and evolve it:

```sh
./gen_plummer_sphere --n 1000 --seed 123 --scale 1.0 --mass 1.0 --output plummer_1000.bin
./nbody_direct_serial --input plummer_1000.bin --nsteps 100 --dt 1e-4 --eps 0.05 --mass 1.0 --energy-every 10 --output final_state.bin
```

Generate a uniform ball with Maxwellian velocities. If `--sigma` is negative or omitted, the generator uses the uniform-sphere virial estimate
`sigma^2 = G M / (5 R)`.

```sh
./gen_uniform_ball_maxwell --n 1000 --seed 456 --radius 1.0 --mass 1.0 --output ball_1000.bin
./nbody_direct_serial --input ball_1000.bin --nsteps 100 --dt 1e-4 --eps 0.05 --mass 1.0 --energy-every 10
```

Add `--timing` to print section timings for I/O, force evaluations,
integration, energy diagnostics, and output writing:

```sh
./nbody_direct_serial --input plummer_1000.bin --nsteps 100 --dt 1e-4 --eps 0.05 --energy-every 10 --timing
```

Run both smoke tests:

```sh
make run-smoke
```

## Solver notes

The default time integrator is Kick-Drift-Kick, selected with
`--integrator kdk`:

1. compute the initial accelerations before the first step;
2. kick velocities by `dt/2` using the current accelerations;
3. drift positions by `dt` with the half-step velocities;
4. compute accelerations at the new positions;
5. kick velocities by `dt/2` using the updated accelerations.

The previous Drift-Kick-Drift variant is still available with
`--integrator dkd` for numerical and performance comparisons.

The energy check uses the same softened potential as the force law:

```text
U = - sum_{i<j} G m^2 / sqrt(|r_i-r_j|^2 + eps^2)
```

The reported verification metric is

```text
abs(E(t) - E(0)) / max(abs(E(0)), dtype_min_normal)
```

A warning is printed if the maximum observed drift exceeds `--energy-tol` (default `1e-3`). This does not terminate the run, because large drift is often an intentional teaching signal: reduce `dt`, increase `eps`, or inspect the initial conditions.

## Intended optimisation path

The baseline is serial on purpose. Natural extensions are:

- **Pay attention to the data qualifiers, like `const`, `resatrict`, and so on, to let the compiler optimize the code**

- convert `compute_accelerations_direct` into an OpenMP loop without inner-loop
  atomics;
- compare Newton-third-law reuse against thread-private force buffers;  
  when is it convenient, against the price of using atomics for a non-local write?
- split accumulators to shorten the floating-point dependency chain;
- compare scalar `sqrt` with an approximate reciprocal-square-root path and
  verify that energy conservation remains meaningful;
- preserve the SoA layout when adding MPI ring-shift communication;
- can you measure the achieved FLOP/s before and after each change.
- Instrument your code so that you can tie every section and assess their scalability separately, instead of just the total run-time
