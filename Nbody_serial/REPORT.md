# Direct N-body HPC Project Report Draft

This is a living draft.  Results are added as soon as they are produced, then
the prose will be refined for the final submission.

## Official Requirements Tracked

- Direct gravitational N-body with softened potential and open boundaries.
- Second-order leapfrog KDK integrator.
- Direct `O(N^2)` force summation, no tree/FMM/PM method.
- Hybrid MPI + OpenMP implementation:
  - MPI ring-shift over particle blocks.
  - OpenMP parallelization inside each rank without atomics in the inner force
    loop.
- Quantitative correctness check independent of runtime: relative total-energy
  drift.
- Strong and weak scaling plots with speedup and parallel efficiency.
- At least five repetitions per measurement point.
- Profiling/timing evidence for bottlenecks.
- Native versus Singularity/Apptainer container overhead, later in the project.

## Current Implementation State

- Particle storage uses Structure of Arrays.
- `nbody_direct_serial` supports both `--integrator kdk` and the original
  baseline `--integrator dkd`.
- `nbody_mpi_omp` implements KDK with MPI ring-shift and OpenMP local force
  loops.
- Timing sections include force, communication, energy, integration, drift, and
  kick where applicable.

## Correctness Results

Serial KDK and MPI KDK were compared on the same Plummer initial condition with
`N = 128`, `nsteps = 5`, `dt = 1e-4`, and `eps = 0.05`.

| Backend | Ranks | Threads | Final total energy | Max relative drift | Status |
|:---|---:|---:|---:|---:|:---|
| Serial KDK | 1 | 1 | -1973.0324198012868 | 1.401851e-09 | OK |
| MPI KDK | 1 | 1 | -1973.0324198012868 | 1.401851e-09 | OK |
| MPI KDK | 2 | 1 | -1973.0324198012868 | 1.401851e-09 | OK |
| MPI KDK | 4 | 1 | -1973.0324198012868 | 1.401851e-09 | OK |

The MPI ring-shift implementation preserves the serial energy diagnostic for
this validation case.

## Hybrid MPI + OpenMP Mapping

First hybrid benchmark on Orfeo EPYC with `N = 8192`, `nsteps = 10`,
`dt = 1e-4`, `eps = 0.05`, five repetitions, and eight total workers:

| Ranks | Threads | Workers | Total median s | Force median s | Comm median s | Speedup vs 1x8 | Status |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 8 | 8 | 0.615687 | 0.523890 | 0.000000 | 1.000 | OK |
| 2 | 4 | 8 | 0.631765 | 0.526246 | 0.028872 | 0.975 | OK |
| 4 | 2 | 8 | 0.629637 | 0.519183 | 0.067801 | 0.978 | OK |
| 8 | 1 | 8 | 0.574912 | 0.496245 | 0.056987 | 1.071 | OK |

At this size, the MPI-heavy `8x1` mapping is the fastest configuration among
the tested layouts, despite nonzero ring communication.

## Native Weak Scaling Pilot

Orfeo EPYC, MPI-only, `Nlocal = 4096`, `nsteps = 20`, five repetitions:

| Ranks | N | Nlocal | Total median s | Force median s | Comm median s | Algorithm-aware weak efficiency | Status |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 4096 | 4096 | 1.969 | 1.889 | 0.000 | 1.000 | OK |
| 2 | 8192 | 4096 | 4.007 | 3.775 | 0.005 | 0.983 | OK |
| 4 | 16384 | 4096 | 8.092 | 7.548 | 0.288 | 0.973 | OK |
| 8 | 32768 | 4096 | 16.251 | 15.105 | 0.847 | 0.969 | OK |

For direct N-body weak scaling, each rank keeps `Nlocal` home particles but
interacts with all `P * Nlocal` particles, so the ideal runtime grows linearly
with `P`.  The normalized weak efficiency is therefore computed as
`P * T1 / TP`.

## Native Strong Scaling Pilot

Orfeo EPYC, MPI-only, `N = 32768`, `nsteps = 20`, five repetitions:

| Ranks | Nlocal | Total median s | Force median s | Comm median s | Speedup | Parallel efficiency | Status |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 32768 | 125.993 | 120.924 | 0.000 | 1.000 | 1.000 | OK |
| 2 | 16384 | 63.968 | 60.316 | 0.074 | 1.970 | 0.985 | OK |
| 4 | 8192 | 32.267 | 30.138 | 1.156 | 3.904 | 0.976 | OK |
| 8 | 4096 | 16.267 | 15.108 | 0.852 | 7.745 | 0.968 | OK |

The strong scaling is close to ideal up to 8 ranks.  The communication cost
increases with rank count but remains smaller than the force time at this
problem size.

## Next Results To Add

- Larger strong scaling, closer to the suggested `N = 10^5`.
- Larger weak scaling, closer to the suggested `Nlocal = 10^4`.
- Hybrid strong/weak scaling with selected rank/thread mappings.
- Vectorization report and profiling counters or code-instrumented FLOP/s.
- Newton's third law trade-off, reciprocal square root, FMA dependency-chain
  tests, and communication-computation overlap.
- Container native-vs-Singularity measurements.
