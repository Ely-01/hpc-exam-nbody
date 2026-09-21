# Direct N-body HPC Exam Repository

This repository contains the implementation, benchmark scripts, container
recipes, and report artifacts for the direct softened gravitational N-body
exercise.

The scientific discussion, interpretation of the results, and final exam
deliverable are in:

```text
REPORT.md
```

This README is a practical guide to the repository: what each file does, which
commands build and run the code, where raw results are written, and how the
tables and figures used in the report are generated.

## Repository Map

```text
Nbody_serial/
├── nbody_common.h
│   └── shared particle format, precision type, math helpers, binary I/O constants
│
├── nbody_core.h, nbody_core.c
│   └── shared SoA particle storage, allocation, binary I/O, energy routines
│
├── nbody_direct_serial.c
│   └── serial/OpenMP solver used for validation and kernel trade-off studies
│
├── nbody_mpi_omp.c
│   └── MPI + OpenMP production solver with blocking/overlap ring communication
│
├── generate_ic.c
│   └── Plummer and uniform-sphere initial-condition generator
│
├── inspect_particles.c
│   └── utility for inspecting binary particle files
│
├── benchmark_layout.c
│   └── AoS-vs-SoA force-kernel layout microbenchmark
│
├── benchmark_rsqrt_kernel.c
│   └── isolated SIMD reciprocal-square-root microbenchmark
│
├── Makefile
│   └── native, MPI, OpenMP, precision, and benchmark builds
│
├── container/
│   ├── Dockerfile
│   │   └── Docker build environment required by the assignment
│   └── nbody.def
│       └── Singularity definition file for the Orfeo .sif image
│
├── scripts/
│   ├── benchmark/
│   │   └── repeated benchmark drivers producing raw CSV files
│   ├── slurm/
│   │   └── Orfeo SLURM wrappers for the benchmark drivers
│   ├── analyze/
│   │   └── CSV summarization, report tables, and SVG plots
│   ├── smoke/
│   │   └── short native/MPI/container smoke tests
│   └── utils/
│       └── system information, MPI binding checks, launch-overhead measurement
│
├── report/
│   ├── data/
│   │   └── hardware and software stack snapshot used in the report
│   ├── tables/
│   │   └── final CSV and Markdown tables used in REPORT.md
│   └── figures/
│       └── final SVG plots used in REPORT.md
│
└── results/
    └── raw benchmark logs and CSV files generated during runs
```

## Build Configuration

Important Makefile variables:

```text
PRECISION=double|float   arithmetic type used by the solvers
OPENMP=0|1               enable OpenMP compilation
CFLAGS=...               optimization and warning flags
MPICC=...                MPI compiler wrapper
```

Native final-report flags on Orfeo:

```sh
CFLAGS="-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic"
```

Portable container benchmark flags:

```sh
CFLAGS="-O3 -march=x86-64-v3 -ffp-contract=fast -Wall -Wextra -Wpedantic"
```

## Build Commands

Build the serial/OpenMP-capable programs:

```sh
make clean
make OPENMP=1 PRECISION=double \
  CFLAGS="-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic"
```

Build the MPI + OpenMP solver:

```sh
make mpi OPENMP=1 PRECISION=double \
  CFLAGS="-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic"
```

Build the isolated reciprocal-square-root microbenchmark:

```sh
make benchmark_rsqrt_kernel PRECISION=double \
  CFLAGS="-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic"
```

Clean generated executables and temporary binary inputs:

```sh
make clean
```

## Initial Conditions and File Format

Generate a Plummer sphere:

```sh
./generate_ic \
  --model 0 \
  --n 10000 \
  --seed 123 \
  --scale 1.0 \
  --mass 1.0 \
  --output results/plummer_10000.bin
```

Generate a uniform ball:

```sh
./generate_ic \
  --model 1 \
  --n 10000 \
  --seed 123 \
  --scale 1.0 \
  --mass 1.0 \
  --output results/uniform_10000.bin
```

All solvers use the same compact binary format:

```text
byte 0..7       magic: "NBODYF1\0"
next 8 bytes    uint64_t particle count N
then N records  x y z vx vy vz, six float values per particle
```

The solver mass is passed at runtime through `--mass`; the file stores positions
and velocities only.

## Direct Solver Runs

Run the serial/OpenMP solver:

```sh
OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
./nbody_direct_serial \
  --input results/plummer_10000.bin \
  --nsteps 100 \
  --dt 1e-4 \
  --eps 0.05 \
  --energy-every 10 \
  --timing
```

Select the force kernel:

```sh
./nbody_direct_serial --input results/plummer_8192.bin --nsteps 20 \
  --dt 1e-4 --eps 0.05 --energy-every 20 \
  --force-kernel direct --timing --quiet

./nbody_direct_serial --input results/plummer_8192.bin --nsteps 20 \
  --dt 1e-4 --eps 0.05 --energy-every 20 \
  --force-kernel direct-split4 --timing --quiet

./nbody_direct_serial --input results/plummer_8192.bin --nsteps 20 \
  --dt 1e-4 --eps 0.05 --energy-every 20 \
  --force-kernel newton-atomic --timing --quiet
```

Select the inverse-square-root path:

```sh
./nbody_direct_serial --input results/plummer_8192.bin --nsteps 20 \
  --dt 1e-4 --eps 0.05 --energy-every 20 \
  --inv-sqrt libm --timing --quiet

./nbody_direct_serial --input results/plummer_8192.bin --nsteps 20 \
  --dt 1e-4 --eps 0.05 --energy-every 20 \
  --inv-sqrt rsqrt2 --timing --quiet
```

## MPI + OpenMP Solver Runs

Run the distributed solver with blocking ring communication:

```sh
mpirun -np 4 ./nbody_mpi_omp \
  --input results/plummer_32768.bin \
  --nsteps 100 \
  --dt 1e-4 \
  --eps 0.05 \
  --energy-every 100 \
  --ring-mode blocking \
  --timing
```

Run the non-blocking overlap variant:

```sh
mpirun -np 4 ./nbody_mpi_omp \
  --input results/plummer_32768.bin \
  --nsteps 100 \
  --dt 1e-4 \
  --eps 0.05 \
  --energy-every 100 \
  --ring-mode overlap \
  --timing
```

`blocking` uses `MPI_Sendrecv`. `overlap` posts `MPI_Irecv`/`MPI_Isend` before
computing the current ring block and waits before swapping buffers.

## Smoke Tests

Native serial smoke test:

```sh
make run-smoke
```

MPI smoke test:

```sh
sh scripts/smoke/run_mpi_smoke.sh
```

Container smoke test:

```sh
CONTAINER_RUNTIME=singularity \
CONTAINER_IMAGE=container/nbody_latest.sif \
sh scripts/smoke/run_container_smoke.sh
```

Equivalent SLURM wrappers:

```sh
sbatch scripts/slurm/mpi_smoke.slurm
sbatch scripts/slurm/container_smoke.slurm
```

## Orfeo Module Setup

Typical native setup:

```sh
module purge
module load openMPI/4.1.6
```

Container setup:

```sh
module purge
module load openMPI/4.1.6 singularity/4.3.1
```

Collect hardware/software information:

```sh
sbatch -A dssc -p GENOA --ntasks=1 --cpus-per-task=1 \
  scripts/slurm/system_info_genoa.slurm
```

Final snapshot:

```text
report/data/system_info_genoa.txt
```

## Validation and Growth Checks

Energy-conservation validation:

```sh
N=10000 NSTEPS=100 DT=1e-4 EPS=0.05 ENERGY_EVERY=10 THREADS=8 \
  sh scripts/benchmark/benchmark_validation.sh
```

SLURM version:

```sh
sbatch -A dssc -p GENOA --ntasks=1 --cpus-per-task=8 \
  --export=ALL,MODULES="openMPI/4.1.6",THREADS=8 \
  scripts/slurm/validation.slurm
```

Outputs:

```text
report/tables/validation_energy_summary.csv
report/tables/validation_energy_summary.md
```

Growth check for the direct `O(N^2)` cost:

```sh
VALIDATION_MODE=growth N_BASE=1000 N_FACTOR=10 NSTEPS=50 \
ENERGY_EVERY=50 THREADS=1 \
  sh scripts/benchmark/benchmark_validation.sh
```

Outputs:

```text
report/tables/n_growth_summary.csv
report/tables/n_growth_summary.md
```

## Kernel-Level Benchmarks

Newton third-law trade-off:

```sh
THREADS="1 2 4 8" REPEATS=5 N=8192 NSTEPS=20 DT=1e-4 EPS=0.05 \
  sh scripts/benchmark/benchmark_newton_tradeoff.sh

python3 scripts/analyze/analyze_newton_tradeoff.py results/newton_tradeoff_*.csv \
  --csv report/tables/newton_tradeoff_summary.csv \
  --markdown report/tables/newton_tradeoff_summary.md \
  --svg report/figures/newton_tradeoff.svg
```

AoS-vs-SoA layout trade-off:

```sh
THREADS="1 2 4 8" REPEATS=5 N=16384 \
  sh scripts/benchmark/benchmark_layout_tradeoff.sh

python3 scripts/analyze/analyze_layout_tradeoff.py results/layout_tradeoff_*.csv \
  --csv report/tables/layout_tradeoff_summary.csv \
  --markdown report/tables/layout_tradeoff_summary.md \
  --svg report/figures/layout_tradeoff.svg
```

Compiler vectorization report for the layout benchmark:

```sh
make OPENMP=1 PRECISION=double benchmark_layout \
  CFLAGS="-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic -fopt-info-vec-all=report/tables/layout_vec_all.txt"
```

Full-solver reciprocal-square-root trade-off:

```sh
THREADS="1 2 4 8" REPEATS=5 N=8192 NSTEPS=20 DT=1e-4 EPS=0.05 \
  sh scripts/benchmark/benchmark_rsqrt_tradeoff.sh

python3 scripts/analyze/analyze_rsqrt_tradeoff.py results/rsqrt_tradeoff_*.csv \
  --csv report/tables/rsqrt_tradeoff_summary.csv \
  --markdown report/tables/rsqrt_tradeoff_summary.md
```

Isolated SIMD reciprocal-square-root microbenchmark:

```sh
REPEATS=5 N=16777216 \
  sh scripts/benchmark/benchmark_rsqrt_kernel.sh

python3 scripts/analyze/analyze_rsqrt_kernel.py results/rsqrt_kernel_*.csv \
  --csv report/tables/rsqrt_kernel_summary.csv \
  --markdown report/tables/rsqrt_kernel_summary.md \
  --svg report/figures/rsqrt_kernel.svg
```

Accumulator-splitting benchmark:

```sh
THREADS="1 2 4 8" REPEATS=5 N=8192 NSTEPS=20 DT=1e-4 EPS=0.05 \
  sh scripts/benchmark/benchmark_accumulator_tradeoff.sh

python3 scripts/analyze/analyze_accumulator_tradeoff.py results/accumulator_tradeoff_*.csv \
  --csv report/tables/accumulator_tradeoff_summary.csv \
  --markdown report/tables/accumulator_tradeoff_summary.md \
  --svg report/figures/accumulator_tradeoff.svg
```

## Hybrid Mapping and Ring Overlap

Hybrid MPI/OpenMP mapping:

```sh
CONFIGS="2x32 8x8 64x1" REPEATS=5 N=32768 NSTEPS=20 \
  sh scripts/benchmark/benchmark_hybrid.sh

python3 scripts/analyze/summarize_hybrid_csv.py results/hybrid_benchmark_*.csv \
  --csv report/tables/hybrid_mapping_summary.csv \
  --markdown report/tables/hybrid_mapping_summary.md
```

Blocking-vs-overlap ring comparison:

```sh
RING_MODE=blocking CONFIGS="4x1 8x1 16x1 32x1" REPEATS=5 N=32768 NSTEPS=20 \
  sh scripts/benchmark/benchmark_hybrid.sh

RING_MODE=overlap CONFIGS="4x1 8x1 16x1 32x1" REPEATS=5 N=32768 NSTEPS=20 \
  sh scripts/benchmark/benchmark_hybrid.sh

python3 scripts/analyze/analyze_ring_overlap.py \
  results/hybrid_benchmark_blocking_*.csv \
  results/hybrid_benchmark_overlap_*.csv \
  --csv report/tables/ring_overlap_summary.csv \
  --markdown report/tables/ring_overlap_summary.md \
  --svg report/figures/ring_overlap.svg
```

## Native Strong and Weak Scaling

Native-only scaling uses the host MPI default transport, because it measures
native node/application performance rather than container overhead isolation.

Strong scaling on Orfeo:

```sh
sbatch -A dssc -p GENOA --ntasks=32 --cpus-per-task=1 --time=02:00:00 \
  --export=ALL,MODULES="openMPI/4.1.6",MODE=strong,BACKEND=native,N=32768,NSTEPS=100,REPEATS=5,CONFIGS="1x1 2x1 4x1 8x1 16x1 32x1",CSV=results/final_native_strong_default.csv \
  scripts/slurm/scaling_benchmark.slurm
```

Weak scaling on Orfeo:

```sh
sbatch -A dssc -p GENOA --ntasks=16 --cpus-per-task=1 --time=02:00:00 \
  --export=ALL,MODULES="openMPI/4.1.6",MODE=weak,BACKEND=native,NLOCAL=8192,NSTEPS=100,REPEATS=5,CONFIGS="1x1 2x1 4x1 8x1 16x1",CSV=results/final_native_weak_default.csv \
  scripts/slurm/scaling_benchmark.slurm
```

Generate native scaling tables and figures:

```sh
python3 scripts/analyze/summarize_scaling_csv.py \
  results/final_native_strong_default.csv \
  --csv report/tables/strong_scaling_native_summary.csv \
  --markdown report/tables/strong_scaling_native_summary.md

python3 scripts/analyze/plot_scaling_svg.py \
  report/tables/strong_scaling_native_summary.csv \
  --view native \
  --output report/figures/strong_scaling_native_final.svg

python3 scripts/analyze/summarize_scaling_csv.py \
  results/final_native_weak_default.csv \
  --csv report/tables/weak_scaling_native_summary.csv \
  --markdown report/tables/weak_scaling_native_summary.md

python3 scripts/analyze/plot_scaling_svg.py \
  report/tables/weak_scaling_native_summary.csv \
  --view native \
  --output report/figures/weak_scaling_native_final.svg
```

## Container Workflow

The repository contains both the Dockerfile requested by the assignment and the
Singularity definition file used for Orfeo:

```text
container/Dockerfile
container/nbody.def
```

Build the `.sif` image from the definition file:

```sh
module load singularity/4.3.1
singularity build container/nbody_latest.sif container/nbody.def
```

The generated `.sif` is not committed.

Check MPI library binding:

```sh
module load openMPI/4.1.6 singularity/4.3.1
make mpi OPENMP=1 PRECISION=double \
  CFLAGS="-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic"

CONTAINER_RUNTIME=singularity \
CONTAINER_IMAGE=container/nbody_latest.sif \
sh scripts/utils/check_container_mpi_binding.sh
```

Expected outputs:

```text
results/mpi_ldd_native.txt
results/mpi_ldd_container_build_env.txt
results/mpi_ldd_container_host_mpi.txt
```

Measure launch overhead:

The utility option is named `--apptainer` for historical compatibility, but the
runtime passed on Orfeo is `singularity`.

```sh
python3 scripts/utils/measure_container_launch_overhead.py \
  --apptainer singularity \
  --image container/nbody_latest.sif \
  --repeats 10 \
  --output results/container_launch_overhead.csv \
  --csv report/tables/container_launch_overhead_summary.csv \
  --markdown report/tables/container_launch_overhead_summary.md
```

## Native-vs-Container Scaling

The final native-vs-container comparison uses a controlled MPI transport policy
for both backends:

```text
MPI_POLICY=self_tcp
```

Internally this sets:

```text
OMPI_MCA_pml=^ucx
OMPI_MCA_btl=self,tcp
OMPI_MCA_osc=^ucx
OMPI_MCA_btl_vader_single_copy_mechanism=none
```

Strong scaling, controlled native:

```sh
sbatch -A dssc -p GENOA --ntasks=32 --cpus-per-task=1 --time=02:00:00 \
  --export=ALL,MODULES="openMPI/4.1.6",MODE=strong,BACKEND=native,EXPORT_MPI_MCA=1,MPI_POLICY=self_tcp,N=32768,NSTEPS=100,REPEATS=5,CONFIGS="1x1 2x1 4x1 8x1 16x1 32x1",CSV=results/final_controlled_native_strong.csv \
  scripts/slurm/scaling_benchmark.slurm
```

Strong scaling, controlled container:

```sh
sbatch -A dssc -p GENOA --ntasks=32 --cpus-per-task=1 --time=02:00:00 \
  --export=ALL,MODULES="openMPI/4.1.6 singularity/4.3.1",MODE=strong,BACKEND=container,CONTAINER_RUNTIME=singularity,CONTAINER_IMAGE=container/nbody_latest.sif,EXPORT_MPI_MCA=1,MPI_POLICY=self_tcp,N=32768,NSTEPS=100,REPEATS=5,CONFIGS="1x1 2x1 4x1 8x1 16x1 32x1",CSV=results/final_controlled_container_strong.csv \
  scripts/slurm/scaling_benchmark.slurm
```

Weak scaling, controlled native:

```sh
sbatch -A dssc -p GENOA --ntasks=16 --cpus-per-task=1 --time=02:00:00 \
  --export=ALL,MODULES="openMPI/4.1.6",MODE=weak,BACKEND=native,EXPORT_MPI_MCA=1,MPI_POLICY=self_tcp,NLOCAL=8192,NSTEPS=100,REPEATS=5,CONFIGS="1x1 2x1 4x1 8x1 16x1",CSV=results/final_controlled_native_weak.csv \
  scripts/slurm/scaling_benchmark.slurm
```

Weak scaling, controlled container:

```sh
sbatch -A dssc -p GENOA --ntasks=16 --cpus-per-task=1 --time=02:00:00 \
  --export=ALL,MODULES="openMPI/4.1.6 singularity/4.3.1",MODE=weak,BACKEND=container,CONTAINER_RUNTIME=singularity,CONTAINER_IMAGE=container/nbody_latest.sif,EXPORT_MPI_MCA=1,MPI_POLICY=self_tcp,NLOCAL=8192,NSTEPS=100,REPEATS=5,CONFIGS="1x1 2x1 4x1 8x1 16x1",CSV=results/final_controlled_container_weak.csv \
  scripts/slurm/scaling_benchmark.slurm
```

Generate controlled native-vs-container tables and figures:

```sh
python3 scripts/analyze/summarize_scaling_csv.py \
  results/final_controlled_native_strong.csv \
  results/final_controlled_container_strong.csv \
  --csv report/tables/strong_scaling_native_container_summary.csv \
  --markdown report/tables/strong_scaling_native_container_summary.md

python3 scripts/analyze/plot_scaling_svg.py \
  report/tables/strong_scaling_native_container_summary.csv \
  --view native-vs-container \
  --output report/figures/strong_scaling_native_container_final.svg

python3 scripts/analyze/summarize_scaling_csv.py \
  results/final_controlled_native_weak.csv \
  results/final_controlled_container_weak.csv \
  --csv report/tables/weak_scaling_native_container_summary.csv \
  --markdown report/tables/weak_scaling_native_container_summary.md

python3 scripts/analyze/plot_scaling_svg.py \
  report/tables/weak_scaling_native_container_summary.csv \
  --view native-vs-container \
  --output report/figures/weak_scaling_native_container_final.svg

python3 scripts/analyze/container_overhead_table.py \
  report/tables/strong_scaling_native_container_summary.csv \
  report/tables/weak_scaling_native_container_summary.csv \
  --csv report/tables/container_overhead_summary.csv \
  --markdown report/tables/container_overhead_summary.md
```

## OSU MPI Microbenchmark

OSU Micro-Benchmarks are not committed. They can be built in user space on
Orfeo and used to generate raw latency/bandwidth output files. The analyzer
expects repeated native and container OSU outputs and produces both a compact
summary and full curves:

```sh
python3 scripts/analyze/analyze_osu_microbench.py \
  --native-latency report/tables/osu_runs/osu_latency_native_1.txt report/tables/osu_runs/osu_latency_native_2.txt report/tables/osu_runs/osu_latency_native_3.txt report/tables/osu_runs/osu_latency_native_4.txt report/tables/osu_runs/osu_latency_native_5.txt \
  --container-latency report/tables/osu_runs/osu_latency_container_1.txt report/tables/osu_runs/osu_latency_container_2.txt report/tables/osu_runs/osu_latency_container_3.txt report/tables/osu_runs/osu_latency_container_4.txt report/tables/osu_runs/osu_latency_container_5.txt \
  --native-bandwidth report/tables/osu_runs/osu_bw_native_1.txt report/tables/osu_runs/osu_bw_native_2.txt report/tables/osu_runs/osu_bw_native_3.txt report/tables/osu_runs/osu_bw_native_4.txt report/tables/osu_runs/osu_bw_native_5.txt \
  --container-bandwidth report/tables/osu_runs/osu_bw_container_1.txt report/tables/osu_runs/osu_bw_container_2.txt report/tables/osu_runs/osu_bw_container_3.txt report/tables/osu_runs/osu_bw_container_4.txt report/tables/osu_runs/osu_bw_container_5.txt \
  --latency-size 1 \
  --bandwidth-size 4194304 \
  --csv report/tables/mpi_microbenchmark_summary.csv \
  --markdown report/tables/mpi_microbenchmark_summary.md \
  --curve-csv report/tables/mpi_microbenchmark_curve.csv \
  --curve-markdown report/tables/mpi_microbenchmark_curve.md \
  --svg report/figures/mpi_microbenchmark_curve.svg
```

The raw `osu_runs/` files are useful for reproducibility but do not need to be
tracked if they are kept only on Orfeo.

## Output Locations

Raw benchmark output:

```text
results/
```

Report-ready tables:

```text
report/tables/*.csv
report/tables/*.md
```

Report-ready plots:

```text
report/figures/*.svg
```

Hardware/software stack snapshot:

```text
report/data/system_info_genoa.txt
```


## Recommended Final Workflow

1. Build native binaries on Orfeo.
2. Run smoke tests.
3. Collect system information.
4. Run validation and growth checks.
5. Run kernel-level benchmarks.
6. Run native strong/weak scaling.
7. Run controlled native-vs-container scaling.
8. Run launch-overhead and OSU MPI microbenchmarks.
9. Generate `report/tables/` and `report/figures/`.
10. Use `REPORT.md` for the scientific discussion.
