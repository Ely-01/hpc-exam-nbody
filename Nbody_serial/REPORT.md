# HPC & Cloud Computing Report

## Exercise 1: Direct N-body Gravitational Simulation

Course: Data Science & AI    
Student: Elisabeth Imbalzano   
Student ID: SM3800119  
Exam session: 01/10/2026


## Abstract

This project investigates the implementation and performance optimization of a direct gravitational N-body simulation on a modern HPC system. The solver evaluates all pairwise gravitational interactions using a softened potential, resulting in an $O(N^2)$  computational kernel, and advances the system in time with a second-order Kick-Drift-Kick leapfrog integrator.

The implementation combines OpenMP shared-memory parallelism with an MPI ring-shift decomposition, allowing each process to retain ownership of its local particles while progressively exchanging source-particle blocks with the other ranks. Several kernel-level optimization strategies are evaluated, including Newton's third-law reuse, data-layout choices, reciprocal-square-root approximations, accumulator splitting, and communication-computation overlap.

Correctness is assessed independently of execution time through total-energy conservation, while performance is characterized through repeated measurements, strong and weak scaling, hybrid MPI/OpenMP mapping, and instrumented kernel timings. The final implementation is also evaluated inside a Singularity container to quantify launch, communication, and application-level overhead relative to native execution.

All final experiments were performed on the Orfeo HPC cluster using its AMD EPYC GENOA partition, since the originally targeted LEONARDO system was unavailable.


## 1. Introduction
Direct **N-body simulation** models the motion of particles interacting through mutual gravitational forces. Since each particle interacts with every other particle, direct force evaluation requires $O(N^2)$ interactions at each time step, making the problem computationally demanding for large systems.

Although more scalable approaches, such as tree-based methods, can reduce this complexity, this project focuses on the **direct formulation** because its regular computational structure provides a clear setting for studying high-performance computing techniques.

The work is organized around four main aspects:

1. **Numerical correctness**  
Verify that the simulation preserves the physical behaviour of the system through total-energy conservation.
2. **Computational optimization**  
Analyze the main computational bottleneck and evaluate different strategies to improve the performance of the force calculation.
3. **Parallel scalability**  
Use OpenMP and MPI to study shared- and distributed-memory parallelism, communication costs, hybrid configurations, and strong and weak scaling.
4. **Portability and containerization**  
Compare native and Singularity executions to evaluate the performance impact and portability of the containerized application.

The overall goal is to **measure and explain** how implementation, optimization, and runtime choices affect correctness, performance, scalability, and portability.


## 2. Physical Model and Numerical Method
### 2.1 Gravitational N-body Model

The system consists of **$N$ equal-mass particles** moving in three-dimensional space under mutual gravitational attraction and open boundary conditions.

Each particle $i$ is described by:

- **position** $\mathbf{x}_i$
- **velocity** $\mathbf{v}_i$
- **mass** $m$

In all final experiments, the particle mass is fixed to

$$
m=1.0,
$$

so the total mass of the system is $M=N$.

The equations of motion are

$$
 \frac{d\mathbf{x}_i}{dt}=\mathbf{v}_i, \qquad \frac{d\mathbf{v}_i}{dt}=\mathbf{a}_i. 
$$

The acceleration of particle $i$ is computed by summing the gravitational contribution of all the other particles:

$$ 
\mathbf{a}_i = Gm \sum_{j\neq i} \frac{\mathbf{x}_j-\mathbf{x}_i} {\left(|\mathbf{x}_j-\mathbf{x}_i|^2+\epsilon^2\right)^{3/2}}. 
$$

The parameter $\epsilon$ is the **gravitational softening length**.

Its role is to prevent the Newtonian force from becoming singular when two particles get extremely close. Without softening, the force would grow without bound as the distance approaches zero. With $\epsilon>0$, very close encounters remain finite and numerically manageable.

The same softening is also used in the potential energy,

$$ 
V = -\sum_{i<j} \frac{Gm^2} {\sqrt{|\mathbf{x}_i-\mathbf{x}_j|^2+\epsilon^2}}, 
$$

so the force computation and the energy diagnostic remain physically consistent.

### 2.2 Initial Conditions and Choice of Softening

The main initial condition used in the experiments is a **Plummer sphere**, a spherically symmetric self-gravitating particle distribution commonly used as a controlled test case.

The final experiments use:

$$
\text{softening length}\quad\epsilon=0.05, 
\qquad \text{and integration timestep}\quad\Delta t=10^{-4}.
$$

The value of $\epsilon$ controls how strongly close particle encounters are smoothed:

- **small $\epsilon$**
the interaction remains closer to the Newtonian force at short distances, but close encounters can produce very *large accelerations*. These rapid changes in the motion may require a *smaller timestep to maintain numerical stability and good energy conservation*;
- **large $\epsilon$**
the interaction becomes smoother and easier to integrate numerically, but *gravitational forces are weakened over a larger short-range region*. Therefore, increasing $\epsilon$ also *changes the physical dynamics represented by the model*.

A useful reference is the **typical distance between particles**,

$$ 
\ell \sim \frac{R}{N^{1/3}}, 
$$

where $R$ is the characteristic size of the system.

This estimate comes from assuming that $N$ particles occupy a volume of order $R^3$: the average volume per particle is then $R^3/N$, and its characteristic linear size is approximately $(R^3/N)^{1/3}$.

For the validation case with $N=10^4$ and $R\approx1$,

$$ 
\ell \approx 0.046, 
$$

which is close to the chosen value

$$ 
\epsilon=0.05. 
$$

Therefore, the selected softening acts on a scale comparable to the typical particle separation. This provides a practical compromise: it limits excessively strong close encounters without smoothing the gravitational interaction over much larger spatial scales.

Changing $\epsilon$ does **not change** the $O(N^2)$ cost of a single force evaluation, because every particle still interacts with every other particle.  
However, it can affect the **total simulation cost indirectly**. A smaller $\epsilon$ allows stronger close-range accelerations and faster changes in particle trajectories, which may require a smaller timestep to keep the integration stable and preserve energy conservation. Simulating the same physical time would then require *more integration steps and* therefore more $O(N^2)$ *force evaluations*.


### 2.3 Kick-Drift-Kick Leapfrog Integration

The system is advanced in time using a second-order **Kick-Drift-Kick (KDK) leapfrog integrator**.  
Each timestep $\Delta t$ is divided into three stages:

1. **First half kick $-$ update the velocity:**

    The current acceleration is used to advance the velocity by half a timestep

    $$ 
    \mathbf{v}\left(t+\frac{\Delta t}{2}\right) = \mathbf{v}(t) + \frac{\Delta t}{2}\mathbf{a}(t). 
    $$

2. **Drift $-$ update the position:**

    The intermediate velocity is then used to advance the particle positions over a full timestep

    $$ 
    \mathbf{x}(t+\Delta t) = \mathbf{x}(t) + \Delta t\, \mathbf{v}\left(t+\frac{\Delta t}{2}\right).
    $$

    Since the particles have moved, the gravitational ***forces** must be recomputed* at the new positions

    $$ 
    \mathbf{a}(t+\Delta t) = \mathbf{a}\!\left[\mathbf{x}(t+\Delta t)\right]. 
    $$

3. **Second half kick $-$ complete the velocity update:**

    The newly computed acceleration is used to advance the velocity by the remaining half timestep

    $$ 
    \mathbf{v}(t+\Delta t) = \mathbf{v}\left(t+\frac{\Delta t}{2}\right) + \frac{\Delta t}{2}\mathbf{a}(t+\Delta t). 
    $$

After these three stages, both positions and velocities are available at time $t+\Delta t$, and the procedure is repeated for the next timestep.

The KDK leapfrog method is **second-order, symplectic, and time-reversible**. These properties make it particularly suitable for gravitational simulations, because it provides good long-term behaviour and *limits artificial drift* in the system energy.


### 2.4 Energy Conservation as a Correctness Criterion

For an isolated gravitational system, the total mechanical energy should remain constant over time. Energy conservation therefore provides a simple physical criterion for checking that the numerical solver evolves the system correctly, independently of its execution time.

The total energy is
$$
E = T + V,
$$

where the kinetic energy is

$$
T = \frac{1}{2}
\sum_i m |\mathbf{v}_i|^2
$$

and $V$ is the softened gravitational potential defined previously.

Because numerical integration introduces small errors, the energy is *not expected to remain exactly constant*.  
The solver therefore monitors the **relative energy drift**

$$
\delta_E(t)
=
\frac{|E(t)-E(0)|}
{\max(|E(0)|,\mathrm{tiny})},
$$

which measures the relative change in total energy with respect to its initial value.

A small value of $\delta_E$ indicates that the numerical evolution remains close to the expected conservative behaviour.  
The **maximum drift** over the sampled simulation times is used as the **validation metric**.

The final validation run uses:

- $N=10\,000$ particles,
- $100$ KDK integration steps,
- timestep $\Delta t = 10^{-4}$,
- softening length $\epsilon = 0.05$,
- $8$ OpenMP threads,
- energy sampling every $10$ integration steps,
- validation tolerance $\delta_E^{\max}<10^{-4}$.

The measured maximum relative energy drift is

$$
\boxed{
\delta_E^{\max}
=
2.67\times10^{-7}
}
$$

which is well below the adopted validation threshold of $10^{-4}.$

This confirms that the numerical integration preserves the total mechanical energy with a large margin relative to the selected correctness threshold.


## 3. Parallel Implementation

This section describes how the direct N-body force computation is parallelized using **OpenMP**, **MPI**, and a hybrid combination of the two.

### 3.1 Data Layout

Particle data are stored using a Structure-of-Arrays (SoA) representation. Each physical quantity is kept in a separate contiguous array:

```text
x[0 ... N-1]
y[0 ... N-1]
z[0 ... N-1]

vx[0 ... N-1]
vy[0 ... N-1]
vz[0 ... N-1]

ax[0 ... N-1]
ay[0 ... N-1]
az[0 ... N-1]
```

This organization keeps together the values of the same physical quantity. During the force computation, the kernel mainly reads the position arrays and writes the acceleration arrays, without accessing unrelated velocity data.

SoA is also convenient for MPI communication, since particle coordinates are stored in contiguous arrays, and provides a data organization suitable for SIMD-oriented optimizations.

The production implementation uses SoA throughout. A controlled comparison with an Array-of-Structures (AoS) representation is presented later in the optimization experiments.

### 3.2 OpenMP Parallelization

Within a shared-memory node, OpenMP parallelizes the outer loop **over the target particles**.

Conceptually:

```text
parallel for each target particle i:
    ax_i = 0
    ay_i = 0
    az_i = 0

    for each source particle j:
        if i != j:
            compute interaction(i, j)
            accumulate into acceleration of i
```

Each OpenMP iteration is responsible for one target particle $i$. Therefore, **each thread** writes only to the accelerations of the particles assigned to it, while the source-particle positions are shared and read-only.

This makes the iterations independent and **avoids atomic** operations or other synchronization inside the force loop.

The drift and kick stages of the KDK integrator are also parallelized, but their $O(N)$ cost is much smaller than the $O(N^2)$ force computation.

### 3.3 MPI Ring Decomposition

For distributed-memory execution, the $N$ particles are divided into equal contiguous blocks across $P$ MPI ranks.

Each rank owns

$$
N_{\mathrm{local}}=\frac{N}{P}
$$

particles. These particles remain the rank's **local target particles** throughout the simulation.

However, each local target must interact with all $N$ particles in the system. To make every source particle available without storing the complete system on every rank, the implementation uses a ring communication pattern.

Each rank therefore works with:

- a fixed block of **local target particles**;
- a **source block** that moves between MPI ranks.

Initially, each rank uses its own local block as the source block. After computing those interactions, the source block is passed to the next MPI rank and a new block is received from the previous rank.

For $P$ ranks, the procedure is:

```text
for each ring phase:
    compute interactions between
        local targets
        and current source block

    send current source block to next rank
    receive next source block from previous rank
```

After $P$ phases, every rank has seen every source block. Therefore, each local target has interacted with all $N$ particles.

For four MPI ranks:

```text
phase 0:   R0 uses B0   R1 uses B1   R2 uses B2   R3 uses B3
phase 1:   R0 uses B3   R1 uses B0   R2 uses B1   R3 uses B2
phase 2:   R0 uses B2   R1 uses B3   R2 uses B0   R3 uses B1
phase 3:   R0 uses B1   R1 uses B2   R2 uses B3   R3 uses B0
```

The target particles never move between ranks: only the source blocks circulate around the ring.

As a consequence, *each rank remains the unique owner of the accelerations of its local particles, so no global reduction of the acceleration arrays is required* after the ring traversal.

The current implementation assumes that $N$ is divisible by the number of MPI ranks, which is sufficient for all benchmark configurations used in this work.

### 3.4 Hybrid MPI + OpenMP Execution

The MPI ring decomposition and OpenMP parallelization are combined in the hybrid solver.

The two levels have distinct roles:

- **MPI** divides the target particles among ranks and circulates the source blocks;
- **OpenMP** divides the local target particles among the threads of each rank.

Therefore, during each ring phase, a rank receives one source block and its OpenMP threads collaboratively compute the interactions between that block and the rank's local targets.

Conceptually:
```text
MPI rank owns a block of target particles

for each source block in the ring:
    OpenMP threads divide the local targets
    each thread computes interactions with the source block

    rotate the source block to the next MPI rank
```

The total number of cores can be distributed between MPI ranks and OpenMP threads in different ways. For example, the same hardware resources can be used with a small number of ranks and many threads per rank, or with many ranks and fewer threads.

These configurations change the balance between shared-memory computation, MPI communication, and NUMA locality. Their performance is compared experimentally later in the report.

### 3.5 Blocking and Non-Blocking Ring Communication

Two communication strategies are implemented for the MPI ring.

#### $-$ Blocking communication $-$

The reference implementation uses `MPI_Sendrecv`.

For each ring phase, the rank first computes the interactions with its current source block and then exchanges that block with its neighbours:

```text
compute using current source block
            ↓
exchange source blocks
            ↓
compute using next source block
```

The communication and computation therefore occur one after the other.

#### $-$ Non-blocking communication $-$

The non-blocking implementation uses `MPI_Irecv` and `MPI_Isend`.

The main idea is to *start the communication for the next ring phase while the rank is still computing the current one*.

Conceptually:
```text
start transfer of next source block
            ↓
compute using current source block
            ↓
wait only if communication is not finished
            ↓
continue with the received source block
```

Unlike a blocking call, `MPI_Isend` and `MPI_Irecv` return immediately, allowing the rank to continue computing while the data transfer progresses.

If the communication finishes before the current force computation, part or all of its cost is hidden behind useful computation. Before the received block can be used in the next phase, `MPI_Waitall` ensures that the transfer has completed.

The intended execution is therefore:

$$ 
\boxed{ \text{communication} \quad \text{overlaps with} \quad \text{force computation} } 
$$

rather than

$$
\boxed{ \text{force computation} \rightarrow \text{communication} }
$$

as in the blocking version.

However, non-blocking communication *does not guarantee complete overlap*. The actual benefit depends on factors such as message size, MPI progress, communication latency, and the amount of computation available while the transfer is in progress.

The *blocking implementation is therefore used as the reference*, while the non-blocking version is evaluated experimentally to determine how much communication can actually be hidden.


## 4. Experimental Setup and Methodology

All final experiments were performed on the Orfeo HPC cluster, using the `GENOA` CPU partition. The original assignment targeted LEONARDO; however, LEONARDO was not available, so the benchmark campaign was moved to Orfeo while preserving the required experimental methodology.

System information was collected directly inside a SLURM allocation on a GENOA compute node rather than on the login node. This is important because processor topology, NUMA configuration, available memory, and software modules may differ between login and compute nodes.

### 4.1 Hardware Platform

The GENOA node used for the experiments is based on two AMD EPYC 9374F processors.

| Component | Configuration |
|:---|:---|
| CPU | AMD EPYC 9374F 32-Core Processor |
| Sockets | 2 |
| Cores per socket | 32 |
| Hardware threads per core | 1 |
| Physical cores / visible CPUs | 64 |
| SMT | Disabled / not exposed |
| NUMA domains | 8 |
| Cores per NUMA domain | 8 |
| Main memory | 503 GiB |
| Swap | none |

The NUMA layout recorded from the allocated compute node is:

| NUMA node | Logical CPUs | Memory size |
|---:|:---|---:|
| 0 | 0-7 | 64053 MB |
| 1 | 8-15 | 64508 MB |
| 2 | 16-23 | 64508 MB |
| 3 | 24-31 | 64437 MB |
| 4 | 32-39 | 64508 MB |
| 5 | 40-47 | 64508 MB |
| 6 | 48-55 | 64508 MB |
| 7 | 56-63 | 64480 MB |

 The `numactl -H` distance matrix reports distance: 10 within each NUMA domain, 12 between domains on the same socket, and 32 between domains on different sockets.
 
 This topology is relevant for the hybrid MPI/OpenMP experiments, because different rank/thread mappings affect both memory locality and the number of MPI ring participants. For this reason, one rank per socket, one rank per NUMA domain, and one rank per physical core are compared experimentally.


### 4.2 Software Environment

The final native experiments were performed using the following software environment:

| Component                    | Version             |
| :--------------------------- | :------------------ |
| C compiler                   | GCC 14.3.1          |
| MPI implementation           | Open MPI 4.1.6      |
| MPI compiler wrapper         | `mpicc`             |
| OpenMP runtime               | GNU `libgomp`       |
| Container runtime            | SingularityCE 4.3.1 |
| Hardware-locality library    | hwloc 2.12.0        |
| GNU C library                | glibc 2.40          |
| External numerical libraries | none                |

The MPI executable is compiled through `mpicc`, the Open MPI compiler wrapper. It uses the underlying GCC compiler while automatically adding the MPI headers and libraries required for compilation and linking.

The force kernel is implemented directly in C and does not rely on BLAS or other external numerical libraries.

All final numerical experiments use double-precision arithmetic.

### 4.3 Compilation Configuration

The final native benchmarks were built using double precision and OpenMP support. The reference build commands were:

```text
make OPENMP=1 PRECISION=double \
  CFLAGS="-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic"
```
```text
make mpi OPENMP=1 PRECISION=double \
  CFLAGS="-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic"
```

The first command builds the serial/OpenMP executables and microbenchmarks, while the second builds the MPI/OpenMP executable through `mpicc`.

The build configuration used for the final benchmarks is:

- `OPENMP=1`: enables OpenMP support where required;
- `PRECISION=double`: selects double-precision floating-point arithmetic.

The main compiler flags are:

- `-O3`: enables aggressive compiler optimizations;
- `-march=native`: optimizes the generated code for the GENOA processor used in the native experiments;
- `-ffp-contract=fast`: allows floating-point contraction, including fused multiply-add operations where applicable;
- `-Wall -Wextra -Wpedantic`: enable additional compiler warnings.

Depending on the selected target, the Makefile also adds:

- `-std=c11`: selects the C11 language standard;
- `-DNBODY_USE_DOUBLE`: selects the double-precision scalar type;
- `-fopenmp`: enables OpenMP compilation and linking.

#### Optional diagnostic flags

Additional compiler options were used only for specific diagnostic experiments and are not part of the standard benchmark build.

For the AoS/SoA layout experiment, compiler vectorization information was generated using:

```text
-fopt-info-vec-all=report/tables/layout_vec_all.txt
```

This produces a report indicating which loops were vectorized and which could not be vectorized.

No profile-guided or feedback-directed optimization was used.

#### Container compilation target

The containerized benchmarks use the same build configuration, except that:

```text
-march=x86-64-v3
```

is used instead of:

```text
-march=native
```

The reason is portability: `-march=native` generates code specifically for the processor used during compilation, whereas `x86-64-v3` defines a portable baseline that can run on a wider range of modern x86-64 processors.

The main difference is:

- `x86-64-v3`
  - provides a portable modern x86-64 baseline;
  - includes instructions such as AVX2 and FMA;
  - does not assume newer extensions such as AVX-512.
- `-march=native`
  - detects the exact GENOA processor;
  - enables the instruction-set extensions supported by that CPU;
  - allows GCC to apply more CPU-specific tuning and instruction scheduling.

Therefore, the *native* build can potentially exploit more *specialized hardware features*, while the *container* build prioritizes *portability*. Whether these additional features produce a measurable performance benefit depends on how effectively the compiler can use them in the application kernel.


### 4.4 Parallel Execution Configuration

Parallel executions were launched through SLURM using `srun`.

Each configuration is identified by:
- $P$: number of MPI ranks;
- $T$: number of OpenMP threads per rank;
- $W=P\times T$: total number of workers.

For example, a $2\times32$ configuration uses 2 MPI ranks with 32 OpenMP threads per rank, for a total of 64 workers.

The general execution command is

```text
srun -n <ranks> -c <threads> ./nbody_mpi_omp [solver options]
```

where:
- `-n` sets the number of MPI ranks;
- `-c` allocates the CPUs required by each rank.

Since SMT is not exposed on the GENOA node, one allocated CPU corresponds to one physical core.

#### OpenMP thread placement
Within each MPI rank, OpenMP thread placement is controlled using

```text
OMP_NUM_THREADS=<threads>
OMP_PLACES=cores
OMP_PROC_BIND=close
```

with the following roles:
- `OMP_NUM_THREADS` sets the number of threads created by each rank;
- `OMP_PLACES=cores` places threads on physical cores;
- `OMP_PROC_BIND=close` keeps the threads of the same rank close to each other within the allocated cores.

MPI rank placement itself is handled by the SLURM `srun` allocation; no additional `mpirun --bind-to` option is used.

#### Verified hybrid mappings

The actual CPU affinity of each MPI rank was explicitly checked inside the SLURM allocation using taskset `-pc`.

The observed mappings were:
| Configuration | Observed affinity                                | Physical interpretation        |
| :------------ | :----------------------------------------------- | :----------------------------- |
| $2\times32$ | rank 0: CPUs 0-31; rank 1: CPUs 32-63            | one MPI rank per socket        |
| $8\times8$  | one 8-core block per rank: 0-7, 8-15, ..., 56-63 | one MPI rank per NUMA domain   |
| $64\times1$ | one distinct CPU per rank                        | one MPI rank per physical core |

These checks confirm that the three hybrid configurations correspond to the intended hardware mappings:

$$ 2\times32 \rightarrow \text{rank per socket} $$
$$ 8\times8 \rightarrow \text{rank per NUMA domain} $$
$$ 64\times1 \rightarrow \text{rank per core} $$

The numerical MPI rank order is not always identical to the NUMA-domain order, but the observed affinity masks confirm that the physical placement is correct.

#### MPI runtime policy

For the controlled native-versus-container experiments, both executions use the same explicit Open MPI runtime policy so that the communication environment is as comparable as possible. The exact settings are described in Section 8.1.

Native-only experiments instead use the *default host MPI* transport unless otherwise stated.

For every benchmark, the number of MPI ranks, OpenMP threads per rank, and total workers are recorded together with the performance measurements.


### 4.5 Benchmark Configurations

The final benchmark campaign includes a set of experiments designed to evaluate numerical correctness, kernel performance, communication behaviour, and parallel scalability.

#### Main application benchmarks

| Experiment        | Problem size                | Steps | Parallel configuration               |
| :---------------- | :-------------------------- | ----: | :----------------------------------- |
| Energy validation | $N=10\,000$               |   100 | 8 OpenMP threads                     |
| Complexity growth | $N=1\,000,\;10\,000$      |    50 | 1 OpenMP thread                      |
| Hybrid mapping    | $N=32\,768$               |    20 | $2\times32,\;8\times8,\;64\times1$ |
| MPI overlap       | $N=32\,768$               |    20 | 4, 8, 16, 32 MPI ranks               |
| Strong scaling    | $N=32\,768$               |   100 | 1, 2, 4, 8, 16, 32 MPI ranks         |
| Weak scaling      | $N_{\mathrm{local}}=8192$ |   100 | 1, 2, 4, 8, 16 MPI ranks             |

#### Kernel optimization benchmarks
The kernel-level experiments use dedicated configurations chosen to isolate specific implementation effects:

| Experiment | Problem size | Execution type |
|:---|:---|:---|
| Newton's third law | $N=8192$, 20 KDK steps | Full solver |
| Reciprocal square root | $N=8192$, 20 KDK steps | Full solver |
| Accumulator splitting | $N=8192$, 20 KDK steps | Full solver |
| AoS vs SoA layout | $N=16\,384$ | Force-kernel microbenchmark |
| Isolated reciprocal square root | $2^{24}$ scalar values | Operation-level microbenchmark |

The three full-solver optimization experiments use the same $N=8192$ configuration to make their performance results directly comparable. The AoS/SoA and isolated reciprocal-square-root tests instead use dedicated microbenchmarks, since their purpose is to isolate a specific kernel property rather than measure the complete N-body simulation.

#### Common numerical parameters
Unless otherwise stated, the time-integration benchmarks use:

| Parameter                     | Value            |
| :---------------------------- | :--------------- |
| Initial condition             | Plummer sphere   |
| Random seed                   | 123              |
| Plummer scale parameter       | 1.0              |
| Timestep $\Delta t$         | $10^{-4}$      |
| Softening length $\epsilon$ | 0.05             |
| Arithmetic precision          | double precision |

The dedicated correctness test uses the validation criterion 

$$
\delta_E^{max}\le 10^{-4}
$$,   
with energy sampled every 10 integration steps, as described in Section 2.4.
For the performance experiments, energy is evaluated only at the end of the run by setting

```text
--energy-every = nsteps
```

so that the diagnostic introduces minimal additional overhead. The default solver threshold 
```text
--energy-tol 1e-3
```
is used only as a runtime warning threshold in these runs and is not the correctness criterion adopted for validation.

#### Adaptation to the available resources

The scaling problem sizes are smaller than those suggested in the original assignment. The reference configuration proposed approximately

$$
N=10^5
$$

for strong scaling and

$$
N/P=10^4
$$

for weak scaling.

These sizes were not practical under the available Orfeo allocation, where jobs on the GENOA partition were limited to a maximum wall time of two hours.

The final campaign therefore uses:

- $N=32\,768$ for strong scaling;
- $N_{\mathrm{local}}=8192$ for weak scaling.

With 16 MPI ranks, the largest weak-scaling case therefore reaches

$$ 
N=16\times8192=131\,072. 
$$

These reduced sizes preserve the structure of the requested scaling experiments while keeping the complete benchmark campaign feasible within the available resource limits.


### 4.6 Measurement and Statistical Methodology

To obtain reproducible performance measurements, the same methodology is used throughout the benchmark campaign unless otherwise stated.

#### Repeated measurements
- Unless otherwise stated, each benchmark configuration is executed *five times* with the same numerical parameters and parallel setup.
- The **median** execution time is used as the representative value, since it is less sensitive to occasional system noise than the arithmetic mean.
- Run-to-run variability is reported using the **sample standard deviation**.
- No repetitions are removed as outliers, and no dedicated warm-up run is discarded.

#### Timing scope

Compilation and initial-condition generation are performed before the benchmark runs and are therefore excluded from the reported execution times.

The solver provides timing instrumentation for the main execution phases:

| Timing component       | Meaning                                                                                                                          |
| :--------------------- | :-------------------------------------------------------------- |
| **Kick time**          | Velocity-update operations of the KDK integrator                |
| **Drift time**         | Position-update operations of the KDK integrator                |
| **MPI communication**  | Exposed time spent in ring communication; in the non-blocking implementation, this represents the communication time not hidden by concurrent force computation. |
| **Initial force**      | First complete force evaluation, performed before the KDK time-stepping loop |
| **Energy diagnostics** | Computation of kinetic and potential energy for energy-conservation checks   |
| **Force time**         | All gravitational force evaluations, including the initial force and the force evaluations performed during the KDK steps |
| **Integration time**   | Complete KDK time-stepping loop, including kick, drift, per-step force evaluations, and diagnostics performed during integration |
| **Total time**         | Complete timed simulation; the broadest application-level timing              |

These timers are not all mutually exclusive. For example, MPI communication is part of the distributed force evaluation, while kick, drift, and per-step force evaluations occur inside the integration loop. They are therefore used to identify the cost of individual execution phases rather than summed to reconstruct the total runtime.


## 5. Baseline Characterization

Before introducing kernel optimizations and parallel scaling, the baseline solver is characterized from three complementary viewpoints:

1. numerical correctness;
2. expected $O(N^2)$ computational growth;
3. distribution of the runtime among the main execution phases.

These measurements establish the reference behaviour of the implementation before optimization.


### 5.1 Numerical Correctness

$$ \boxed{\text{Is it correct?}} $$

The numerical validation procedure is described in Section 2.4.

For the reference Plummer-sphere test, the measured maximum relative energy drift is

$$
\delta_E^{\max}=2.67\times10^{-7},
$$

well below the adopted validation threshold

$$ 10^{-4}. $$

The baseline implementation therefore satisfies the required energy-conservation criterion and provides a numerically valid reference for the following performance experiments.

### 5.2 Verification of $O(N^2)$ Computational Growth

 $$ \boxed{\text{Does it really scale as }N^2\text{?}} $$

The direct force computation evaluates all ordered particle pairs except self-interactions, giving

$$
N(N-1)
$$

interactions per force evaluation.

To verify the expected quadratic growth experimentally, two problem sizes were executed for 50 integration steps using one OpenMP thread:

|       $N$ | Force time (s) |
| ----------: | -------------: |
|  $1\,000$ |       0.180952 |
| $10\,000$ |      18.124287 |

Increasing $N$ from $1\,000$ to $10\,000$ increases the theoretical number of interactions by

$$
\frac{10000(10000-1)} {1000(1000-1)} = 100.090.
$$

The measured force-time ratio is

$$
\frac{18.124287}{0.180952} \approx 100.16.
$$

Therefore,

$$
\text{theoretical ratio}=100.09, \qquad \text{measured ratio}\approx100.16.
$$

The close agreement confirms that the dominant force computation follows the expected $O(N^2)$ growth.

### 5.3 Runtime Bottleneck

$$ \boxed{\text{Where is the time spent?}} $$

The timing instrumentation introduced in Section 4.6 is used to determine where the baseline execution time is spent.

For the $N=10\,000$, 100-step reference run:

| Component          |         Time (s) | Fraction of total |
| :----------------- | ---------------: | ----------------: |
| Force evaluation   |         4.431348 |             90.2% |
| Energy diagnostics |         0.447601 |              9.1% |
| Other operations   | $\approx0.032$ |               <1% |
| **Total**          |     **4.910674** |          **100%** |

Approximately $90\%$ of the execution time is spent evaluating gravitational forces.

This is consistent with the structure of the solver:

- **force evaluation**: $O(N^2)$, since every target interacts with every source;
- **kick and drift**: $O(N)$, since each particle is updated once;
- **energy diagnostics**: also involve pairwise interactions, but are evaluated only periodically.

The direct force kernel is therefore the main computational bottleneck of the baseline implementation and becomes the primary target of the optimization study in the following section.

## 6. Kernel-Level Optimization Study

The baseline analysis identified the direct force kernel as the dominant computational cost. This section therefore investigates how different implementation choices affect its performance.

Four aspects are studied:
- **Newton's third law** $-$ reduce the number of pair evaluations;
- **AoS vs SoA layout** $-$ evaluate the effect of particle-data organization;
- **reciprocal square root** $-$ investigate the cost of the inverse-distance calculation;
- **accumulator splitting** $-$ reduce dependency chains in force accumulation. 

Each experiment isolates one specific optimization trade-off. The corresponding benchmark configurations are summarized in Section 4.5.


### 6.1 Newton's Third Law

The baseline force kernel evaluates both ordered interactions $(i,j)$ and $(j,i)$. 

Using Newton's third law, each pair can instead be evaluated once,

$$
i<j,
$$

and the equal and opposite force contributions are applied to both particles. This reduces the number of pair evaluations from approximately

$$
N(N-1)
$$

to

$$
\frac{N(N-1)}{2}.
$$

The *potential arithmetic saving* is therefore close to $2\times$.  

The main difficulty is parallel force ownership:
- **direct kernel**: each thread updates only its own target acceleration $\mathbf a_i$;
- **Newton reuse**: each pair updates both $\mathbf a_i$ and $\mathbf a_j$, creating possible write conflicts between threads.  

Three implementations were compared:

| Kernel | Pair traversal | Update strategy |
|:---|:---|:---|
| `direct` | ordered pairs | independent target ownership |
| `newton` | $i<j$ | serial, conflict-free |
| `newton-atomic` | $i<j$ | parallel updates protected by atomics |

The `newton-atomic` implementation is included as a *diagnostic case* to measure the cost of a simple synchronization-based solution rather than as a production kernel.

The experiment uses $N=8192$, 20 KDK steps, and five repetitions per configuration.

| Threads | `direct` (s) | `newton` (s) | `newton-atomic` (s) |
|---:|---:|---:|---:|
| 1 | $5.865157 \pm 0.008173$ | $3.510821 \pm 0.003258$ | $8.407843 \pm 0.011617$ |
| 2 | $3.119743 \pm 0.159910$ | $3.553018 \pm 0.022864$ | $8.587850 \pm 0.074676$ |
| 4 | $1.734899 \pm 0.036081$ | $3.510410 \pm 0.000948$ | $7.214137 \pm 0.037040$ |
| 8 | $0.893647 \pm 0.002144$ | $3.520837 \pm 0.025837$ | $6.265592 \pm 0.018730$ |

The `newton` kernel is serial, so its runtime is expected to remain approximately constant when the requested OpenMP thread count changes.

![Newton third-law trade-off](report/figures/newton_tradeoff.svg)

#### Main observations

- **Serial Newton reuse is effective.**  
  With one thread, the force time decreases from $5.87$ s to $3.51$ s:
  $$
  \frac{5.865157}{3.510821}\approx1.67\times.
  $$
  The speedup is lower than the ideal $2\times$ because only the pair-evaluation work is halved; loop overhead, memory operations, and force updates remain.
- **The direct kernel scales much better with OpenMP.**  
  Its runtime decreases from $5.87$ s with one thread to $0.89$ s with eight threads.
- **The serial Newton kernel does not benefit from additional threads.**  
  Its runtime remains close to $3.5$ s, so the direct kernel becomes faster from two threads onward.
- **Atomic conflict resolution is too expensive.**  
  `newton-atomic` is slower than both alternatives, because the synchronization and cache-coherence cost of repeated atomic updates outweighs the arithmetic saved by halving the pair count.

At eight threads:
$$
T_{\mathrm{direct}}=0.89\ \mathrm{s},
\qquad
T_{\mathrm{newton}}=3.52\ \mathrm{s},
\qquad
T_{\mathrm{atomic}}=6.27\ \mathrm{s}.
$$

The direct implementation is therefore approximately $3.9\times$ faster than serial Newton and about $7\times$ faster than the atomic variant.

#### Conclusion

The experiment highlights a key trade-off:

$$
\boxed{
\text{reducing arithmetic work does not necessarily improve parallel performance}
}
$$

Newton's third law is beneficial in the serial case, but the direct target-ownership strategy is better suited to the current OpenMP implementation because it avoids shared writes and synchronization.

For this reason, the production kernel retains the direct formulation. A parallel Newton implementation would require a more efficient conflict-resolution strategy, such as thread-private buffers or block-wise reductions.


### 6.2 AoS versus SoA

The experiment compares two representations of the same particle data:

```text
AoS:
particle[i] = {x, y, z, vx, vy, vz, ax, ay, az}
```
```text
SoA:
x[]  y[]  z[]
vx[] vy[] vz[]
ax[] ay[] az[]
```

In **AoS**, all quantities belonging to one particle are stored together, while in **SoA** each physical quantity is stored in a separate contiguous array.

The production solver uses SoA. To isolate the effect of data organization, the same force computation was benchmarked with both layouts.

The experiment uses $N=16\,384$ and 1, 2, 4, and 8 OpenMP threads. Force times are reported as median $\pm$ sample standard deviation over five repetitions.

The SoA speedup is defined as

$$
S_{\mathrm{SoA}}
=
\frac{T_{\mathrm{AoS}}}{T_{\mathrm{SoA}}},
$$
so $S_{\mathrm{SoA}}>1$ favors SoA, while $S_{\mathrm{SoA}}<1$ favors AoS.

| Threads | AoS force time (s) | SoA force time (s) | SoA speedup |
|---:|---:|---:|---:|
| 1 | $0.903034 \pm 0.000151$ | $0.910895 \pm 0.000439$ | 0.991 |
| 2 | $0.451970 \pm 0.000151$ | $0.455871 \pm 0.001764$ | 0.991 |
| 4 | $0.226311 \pm 0.000205$ | $0.227897 \pm 0.001877$ | 0.993 |
| 8 | $0.115289 \pm 0.001032$ | $0.115800 \pm 0.025404$ | 0.996 |


![AoS versus SoA layout trade-off](report/figures/layout_tradeoff.svg)

#### Main observations
- **No measurable SoA speedup is observed.**  
  The median difference remains below approximately $1\%$ for every thread count.
- **Both layouts scale similarly with OpenMP.**  
  Their force times decrease at nearly the same rate as the number of threads increases.
- **The 8-thread SoA measurement shows higher variability.**  
  Its standard deviation is larger than in the other cases, but the median remains very close to the corresponding AoS result.
- The computed accelerations agree within the numerical tolerance, confirming that the two implementations perform the same calculation.

To *investigate* why SoA does not provide the expected SIMD-related benefit, GCC vectorization diagnostics were inspected. Since hardware-counter tools such as `perf` and PAPI were not available on Orfeo, the analysis relies on measured timings and compiler reports.

The relevant GCC output includes:
```text
benchmark_layout.c:116:29: missed: couldn't vectorize loop
benchmark_layout.c:116:29: missed: not vectorized: unsupported control flow in loop.
nbody_common.h:81:10: missed: statement clobbers memory: ... sqrt(...)

benchmark_layout.c:159:29: missed: couldn't vectorize loop
benchmark_layout.c:159:29: missed: not vectorized: unsupported control flow in loop.
nbody_common.h:81:10: missed: statement clobbers memory: ... sqrt(...)
```

The dominant inner force loop is therefore not effectively vectorized in either layout. The reported obstacles include the self-interaction control flow and the scalar square-root path.

#### Interpretation

The result does **not** imply that AoS and SoA are generally equivalent.  
Rather, it shows that changing the memory layout alone provides little benefit when the computation that should exploit contiguous SoA data is not vectorized.  
In this implementation,

$$
\boxed{
\text{SoA layout alone does not guarantee SIMD speedup}
}
$$

The production solver nevertheless retains SoA because it:
- separates the data streams used by the force kernel;
- provides contiguous arrays convenient for MPI block communication;
- remains a suitable layout for future SIMD-oriented optimization.


### 6.3 Reciprocal Square Root

The gravitational force repeatedly requires the computation of

$$
\frac{1}{\sqrt{x}},
\qquad
x=r^2+\epsilon^2.
$$

The reference implementation evaluates this quantity through the standard mathematical-library square-root path.

A possible alternative is the hardware **reciprocal-square-root estimate** (`rsqrt`), which directly approximates

$$
\frac{1}{\sqrt{x}}.
$$

The hardware estimate is *fast but less accurate*. Its precision can be improved using **Newton-Raphson refinement**. Starting from an approximation

$$
y_k \approx \frac{1}{\sqrt{x}},
$$

one refinement step computes

$$
y_{k+1}
=
y_k
\left(
\frac{3}{2}
-
\frac{1}{2}xy_k^2
\right).
$$

Additional refinement steps generally *improve accuracy*, but also require *extra arithmetic*.

The optimization is evaluated at two levels:
1. **isolated SIMD operation**, to measure the potential of the hardware `rsqrt` instruction;
2. **complete N-body solver**, to determine whether this potential translates into an actual application-level improvement.

#### 6.3.1 Isolated SIMD reciprocal-square-root benchmark
The microbenchmark processes $2^{24}$ independent single-precision values.
The tested methods are:
- `sqrtf`: standard-library single-precision reference;
- `rsqrt0`: raw hardware reciprocal-square-root estimate;
- `rsqrt1`: `rsqrt0` + one Newton refinement;
- `rsqrt2`: `rsqrt0` + two Newton refinements.

The speedup is defined as
$$
S=
\frac{T_{\mathrm{sqrtf}}}{T_{\mathrm{method}}},
$$
so values above 1 indicate faster execution than the sqrtf reference.

| Method | Time (s) | Gvalues/s | Speedup | Max relative error |
|:---|---:|---:|---:|---:|
| `sqrtf` | $0.032575 \pm 0.000021$ | 0.515 | 1.000 | $8.94\times10^{-8}$ |
| `rsqrt0` | $0.001599 \pm 0.000011$ | 10.494 | 20.376 | $2.58\times10^{-4}$ |
| `rsqrt1` | $0.001633 \pm 0.000006$ | 10.277 | 19.953 | $1.72\times10^{-7}$ |
| `rsqrt2` | $0.002469 \pm 0.000004$ | 6.796 | 13.195 | $1.17\times10^{-7}$ |

![SIMD reciprocal-square-root microbenchmark](report/figures/rsqrt_kernel.svg)

#### Main observations
- **`rsqrt0` is extremely fast but less accurate.**  
  It reaches about $20.4\times$ the speed of `sqrtf`, but its maximum relative error increases to
  $$
  2.58\times10^{-4}.
  $$
- **One Newton refinement gives the best speed-accuracy balance.**  
  `rsqrt1` reduces the error to
  $$
  1.72\times10^{-7},
  $$
  while retaining almost the full $20\times$ speedup.
- **The throughput increase is substantial.**  
  The reference sqrtf path processes about $0.515$ Gvalues/s, while `rsqrt0` and `rsqrt1` exceed $10$ Gvalues/s.
- **Further refinement adds cost**.  
  `rsqrt2` slightly improves accuracy, but throughput decreases to $6.796$ Gvalues/s and the speedup falls to about $13.2\times$.

#### Interpretation

The isolated benchmark shows that reciprocal-square-root instructions can provide very high SIMD throughput.  

However, this experiment measures only the mathematical operation itself. It does **not** imply that the complete N-body solver will obtain the same speedup.


#### 6.3.2 Full-solver experiment

The same optimization was then tested inside the complete N-body force kernel.

The reference is the standard **double-precision `sqrt` path provided by `libm`**, consistent with the double-precision solver.

The unrefined `rsqrt0` variant is excluded because its approximation error is too large for the energy-conservation study. The tested variants are:

- `libm`: standard double-precision reference;
- `rsqrt1`: one Newton refinement;
- `rsqrt2`: two refinements;
- `rsqrt3`: three refinements.

The experiment uses $N=8192$, 20 KDK steps, and 8 OpenMP threads.
The solver-level speedup is defined as

$$
S=
\frac{T_{\mathrm{libm}}}{T_{\mathrm{method}}}.
$$

| Method | Force time (s) | Speedup vs `libm` | Max energy drift |
|:---|---:|---:|---:|
| `libm` | $0.894669 \pm 0.055731$ | 1.000 | $4.083561\times10^{-8}$ |
| `rsqrt1` | $1.028367 \pm 0.045048$ | 0.870 | $4.085352\times10^{-8}$ |
| `rsqrt2` | $1.290443 \pm 0.104594$ | 0.693 | $4.083561\times10^{-8}$ |
| `rsqrt3` | $1.476621 \pm 0.010179$ | 0.606 | $4.083561\times10^{-8}$ |

#### Main observations
- **No `rsqrt` variant accelerates the complete solver.**  
  Even `rsqrt1` is slower than the `libm` reference:
  $$
  1.028\ \mathrm{s}
  \quad\text{vs}\quad
  0.895\ \mathrm{s}.
  $$
- **Additional Newton refinements further increase runtime.**  
  The speedup decreases from $0.870$ for `rsqrt1` to $0.606$ for `rsqrt3`.
- **Numerical accuracy is recovered after refinement.**  
  From `rsqrt2` onward, the measured energy drift matches the `libm` reference.

#### Interpretation

The two experiments expose different levels of performance:
- the isolated benchmark shows that **SIMD `rsqrt`** is very fast;
- the full solver shows that **this advantage is not automatically transferred to the application**;
- the refined variants recover the required numerical accuracy, but at additional computational cost.
The key result is

$$
\boxed{
\text{the SIMD rsqrt advantage is lost in the current non-vectorized force kernel}
}
$$

In the current solver, the dominant force loop is not effectively vectorized, so it **cannot exploit** the same SIMD throughput observed in the microbenchmark. At the same time, the `rsqrt` path still pays the cost of the approximation and Newton refinement steps.

Therefore:
- `libm` remains the fastest option for the current full solver;
- refined `rsqrt` variants achieve comparable numerical accuracy;
- `rsqrt` may become advantageous only in a future force kernel that is effectively vectorized.

For this reason, the production implementation retains the standard `libm` path.


### 6.4 Accumulator Splitting and Critical Path

For a fixed target particle, the direct force kernel repeatedly updates the same acceleration components:

$$
a_x \mathrel{+}= \Delta x\,s,
\qquad
a_y \mathrel{+}= \Delta y\,s,
\qquad
a_z \mathrel{+}= \Delta z\,s.
$$

With a single accumulator, each update depends on the result of the previous one:

```text
ax += contribution_0
ax += contribution_1
ax += contribution_2
...
```

This creates a **loop-carried dependency chain**, which limits instruction-level parallelism (ILP).

Accumulator splitting replaces this single chain with several independent partial sums:

```text
ax0 += contribution_0
ax1 += contribution_1
ax2 += contribution_2
ax3 += contribution_3
...
ax = ax0 + ax1 + ax2 + ax3 + ...
```

The number of particle interactions is unchanged. The purpose is only to expose *more independent arithmetic*, allowing the processor to **overlap more operations**.

The benchmark compares 1, 2, 4, and 8 accumulators for each OpenMP thread count. One accumulator corresponds to the original direct kernel.

`Ginteraction/s` measures billions of particle-pair interactions processed per second and is used as the force-kernel throughput metric. The marginal gain measures the improvement relative to the previous accumulator configuration.

| Threads | Accumulators | Force time (s) | Speedup vs direct | Marginal gain | Ginteraction/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | $5.821120 \pm 0.024255$ | 1.000 | - | 0.242 |
| 1 | 2 | $5.659731 \pm 0.003013$ | 1.029 | +2.85% | 0.249 |
| 1 | 4 | $5.697840 \pm 0.003039$ | 1.022 | -0.67% | 0.247 |
| 1 | 8 | $5.656612 \pm 0.008691$ | 1.029 | +0.73% | 0.249 |
| 2 | 1 | $3.094624 \pm 0.208808$ | 1.000 | - | 0.455 |
| 2 | 2 | $2.879254 \pm 0.012164$ | 1.075 | +7.48% | 0.489 |
| 2 | 4 | $2.857590 \pm 0.001465$ | 1.083 | +0.76% | 0.493 |
| 2 | 8 | $2.894601 \pm 0.027718$ | 1.069 | -1.28% | 0.487 |
| 4 | 1 | $1.743009 \pm 0.008529$ | 1.000 | - | 0.808 |
| 4 | 2 | $1.595104 \pm 0.057314$ | 1.093 | +9.27% | 0.883 |
| 4 | 4 | $1.594307 \pm 0.056764$ | 1.093 | +0.05% | 0.884 |
| 4 | 8 | $1.592580 \pm 0.002604$ | 1.094 | +0.11% | 0.885 |
| 8 | 1 | $0.898525 \pm 0.056340$ | 1.000 | - | 1.568 |
| 8 | 2 | $0.805495 \pm 0.030483$ | 1.115 | +11.55% | 1.749 |
| 8 | 4 | $0.796616 \pm 0.029899$ | 1.128 | +1.11% | 1.769 |
| 8 | 8 | $0.780765 \pm 0.022961$ | 1.151 | +2.03% | 1.805 |

![Accumulator splitting critical-path trade-off](report/figures/accumulator_tradeoff.svg)

#### Main observations
- **The first split provides most of the throughput improvement.** 
  Moving from 1 to 2 accumulators improves performance at every thread count. The gain ranges from $2.85\%$ at 1 thread to $11.55\%$ at 8 threads.
- **Additional accumulators give progressively smaller benefits.**  
  From 2 to 4 accumulators, the marginal gain is below $1.2\%$ in all configurations. Increasing from 4 to 8 produces similarly small changes and can even reduce performance.
- **The gain therefore shows clear diminishing returns after the first few accumulators.**  
  For 1-4 threads, performance is essentially saturated by 2-4 accumulators.
- **At 8 threads, split8 gives the best median result, but the extra gain beyond split2 is small.**  
  Throughput increases from $1.749$ Ginteraction/s with 2 accumulators to $1.805$ Ginteraction/s with 8 accumulators. This improvement is much smaller than the initial jump from 1 to 2 accumulators and should also be interpreted in light of the observed run-to-run variability.
- **The largest measured overall improvement occurs at 8 threads with 8 accumulators.**
  $$
  0.898525\ \mathrm{s}
  \rightarrow
  0.780765\ \mathrm{s},
  $$
  corresponding to
  $$
  1.151\times
  $$
  speedup and approximately a $13\%$ reduction in force-kernel runtime.

All tested variants reproduce the same measured energy drift, so the optimization changes performance without affecting the numerical result.

#### Interpretation

The results explain why multiple partial accumulators improve throughput:

```text
1 accumulator
→ one long dependency chain
→ limited ILP

2-4 accumulators
→ several independent chains
→ more arithmetic can overlap
→ higher throughput

further splitting
→ little additional ILP
→ diminishing returns
```

The first few accumulators remove most of the **dependency bottleneck**. After that, the processor already has enough independent arithmetic to keep its execution units busy, so additional accumulators provide little extra benefit.

More accumulators also introduce costs:
- more live registers;
- greater register pressure;
- a final reduction of the partial sums.

The exact best configuration therefore depends on the execution setup, but the overall trend is clear: most of the useful gain is obtained with the first few partial accumulators.

$$
\boxed{
\text{most of the ILP gain is obtained with 2--4 partial accumulators; further splitting gives diminishing returns}
}
$$

Accumulator splitting therefore provides a moderate but measurable optimization, reaching a maximum observed speedup of $1.151\times$ while preserving the numerical result.


#### 6.5 Optimization Summary
- **Newton**: fewer interactions, but more difficult parallel updates.
- **SoA**: suitable layout, but little benefit without vectorization.
- **rsqrt**: high SIMD throughput in isolation, but no solver-level speedup.
- **Accumulator splitting**: best practical improvement, up to $1.151\times$.

Overall, the results show that an optimization is effective only when it matches the structure of the complete kernel.

The next section examines MPI communication and parallel scaling.

## 7. Parallel Performance Analysis

This section evaluates the parallel behaviour of the complete solver.

The analysis focuses on four aspects:

- **MPI/OpenMP balance**: how performance changes with different rank/thread combinations;
- **communication overlap**: whether non-blocking ring communication can hide part of the MPI cost;
- **strong scaling**: how runtime decreases for a fixed problem size;
- **weak scaling**: how performance evolves when the workload per rank is kept constant.

The goal is to determine *whether the application remains compute-bound* as parallelism increases, or whether MPI communication and synchronization *begin to limit scalability*.

### 7.1 Hybrid MPI/OpenMP Mapping

The GENOA node provides 64 physical cores across two sockets and eight NUMA domains. Three hybrid configurations using all 64 cores were compared:

| Configuration | Mapping |
|:---|:---|
| $2\times32$ | one MPI rank per socket |
| $8\times8$ | one MPI rank per NUMA domain |
| $64\times1$ | one MPI rank per physical core |

The experiment uses $N=32768$ and 20 integration steps.

`Ginteraction/s` measures billions of particle interactions processed per second.

| Ranks | Threads | Total time (s) | Force time (s) | Force / total | Ginteraction/s | Communication time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32 | $1.357946 \pm 0.002116$ | 1.208748 | 89.01% | 18.654 | 0.015178 |
| 8 | 8 | $1.356582 \pm 0.009521$ | 1.205893 | 88.89% | 18.698 | 0.094987 |
| 64 | 1 | $1.362196 \pm 0.006402$ | 1.215686 | 89.24% | 18.547 | 0.143501 |

#### Main observations
- **Total runtime is almost unchanged across the three mappings.**  
  The medians differ by only about $0.4\%$.
- **The $8\times8$ configuration has the lowest median runtime**, but the difference is too small relative to the observed variability to identify a clear winner.
- **The force kernel behaves almost identically in all configurations.**  
  It accounts for about $89\%$ of total runtime and sustains approximately
  $$
  18.5\text{--}18.7\ \mathrm{Ginteraction/s}.
  $$
- **Communication cost increases with the number of MPI ranks.**
  $$
  0.015\ \mathrm{s}
  \rightarrow
  0.095\ \mathrm{s}
  \rightarrow
  0.144\ \mathrm{s}
  $$
  for $2$, $8$, and $64$ ranks respectively.

#### Interpretation

Using more MPI ranks increases the number of ring communication phases, while using fewer ranks shifts more parallel work to OpenMP.
However, the communication cost remains small compared with the force computation. Therefore, the increase in MPI communication has little effect on the total runtime for this problem size.

$$
\boxed{
\text{on one GENOA node, the solver remains compute-dominated and the three hybrid mappings perform similarly}
}
$$

The $2\times32$, $8\times8$, and $64\times1$ configurations are therefore all viable on this node. The results show that topology alone is not sufficient to select the best mapping: the actual kernel and problem size must also be considered.


### 7.2 Communication-Computation Overlap

In the blocking ring implementation, each rank performs two separate phases:

```text
force computation
→ communication
```

The non-blocking version attempts to overlap them:
```text
MPI_Irecv / MPI_Isend
        ↓
force computation
        ↓
MPI_Waitall
```

The idea is to start the transfer of the next source block while the current block is still being processed.

The amount of communication hidden by computation is defined as
$$
T_{\mathrm{hidden}}
=
T_{\mathrm{comm,blocking}}
-
T_{\mathrm{comm,overlap}},
$$and the achieved overlap fraction as
$$
f_{\mathrm{overlap}}
=
\frac{T_{\mathrm{hidden}}}
{T_{\mathrm{comm,blocking}}}.
$$

The experiment uses $N=32768$, 20 integration steps, one OpenMP thread per MPI rank, and five repetitions.

| Ranks | Blocking total (s) | Overlap total (s) | Blocking comm. (s) | Hidden comm. (s) | Achieved overlap |
|---:|---:|---:|---:|---:|---:|
| 4 | $20.987808 \pm 0.078594$ | $20.868140 \pm 0.007770$ | 1.036399 | 0.169679 | 16.4% |
| 8 | $10.513724 \pm 0.083738$ | $10.510297 \pm 0.007663$ | 0.678918 | 0.020134 | 3.0% |
| 16 | $5.358677 \pm 0.006079$ | $5.361461 \pm 0.001656$ | 0.478483 | 0.000000 | 0.0% |
| 32 | $2.729329 \pm 0.083506$ | $2.716461 \pm 0.002813$ | 0.226680 | 0.014705 | 6.5% |

![MPI ring communication overlap](report/figures/ring_overlap.svg)

#### Main observations

- **Only a small fraction of communication is actually hidden.**  
  The measured overlap ranges from $0\%$ to $16.4\%$.
- **The highest overlap occurs at 4 ranks.**  
  About $0.170$ s of the $1.036$ s blocking communication time is hidden.
- **At 8 and 32 ranks, the hidden fraction is small, only $3.0\%$ and $6.5\%$, respectively.**  
- **At 16 ranks, no measurable communication is hidden.**
- **The effect on total runtime is negligible.**
  The total-runtime speedup,
  $$
  S=
  \frac{T_{\mathrm{blocking}}}
  {T_{\mathrm{overlap}}},
  $$remains very close to 1:
  $$
  1.006,\quad
  1.000,\quad
  0.999,\quad
  1.005
  $$
  for 4, 8, 16, and 32 ranks.

Therefore, the non-blocking version does not provide a significant application-level speedup in this experiment.

#### Interpretation

Non-blocking MPI calls allow communication and computation to be issued concurrently, but they do not guarantee that communication progresses completely in the background.

In the measured implementation, only a small part of the communication is hidden. Possible exposed costs include:
- MPI progress and message handling;
- synchronization at `MPI_Waitall`;
- communication startup overhead;
- small imbalance between ranks.

More importantly, communication is already a small fraction of the total runtime compared with the $O(N^2)$ force calculation. Therefore, even perfect overlap could only reduce a limited portion of the total execution time.

$$
\boxed{
\text{non-blocking MPI exposes overlap opportunities, but communication is too small to produce a meaningful total-runtime gain}
}
$$

For this one-node workload, blocking and non-blocking ring communication therefore perform almost equivalently at the application level.


### 7.3 Strong Scaling

Strong scaling evaluates how the execution time decreases when more parallel resources are used for a **fixed global problem size**.

The experiment keeps

$$
N=32768
$$

fixed and varies the number of MPI ranks as

$$
P\in\{1,2,4,8,16,32\}.
$$

Each rank uses one OpenMP thread, and every configuration performs 100 KDK integration steps.

The speedup is
$$
S(P)=\frac{T(1)}{T(P)},
$$

where $T(P)$ is the total runtime using $P$ MPI ranks.
Parallel efficiency is

$$
E(P)=\frac{S(P)}{P},
$$

which measures how close the observed speedup is to the ideal linear case.

Ideally,

$$
S(P)=P,
\qquad
E(P)=1.
$$

![Native strong scaling](report/figures/strong_scaling_native_final.svg)

| MPI ranks | Total time (s) | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | $374.387700 \pm 0.490059$ | 1.000 | 1.000 |
| 2 | $187.821448 \pm 0.065840$ | 1.993 | 0.997 |
| 4 | $94.101276 \pm 0.017829$ | 3.979 | 0.995 |
| 8 | $47.148061 \pm 0.005166$ | 7.941 | 0.993 |
| 16 | $23.620087 \pm 0.005129$ | 15.850 | 0.991 |
| 32 | $11.870895 \pm 0.015770$ | 31.538 | 0.986 |


Main observations
- **Runtime almost halves whenever the number of ranks doubles.**  
  $$
  374.4
  \rightarrow
  187.8
  \rightarrow
  94.1
  \rightarrow
  47.1
  \rightarrow
  23.6
  \rightarrow
  11.9\ \mathrm{s}.
  $$

- **Speedup remains very close to ideal.**  
  At 32 ranks,

  $$
  S(32)=31.538,
  $$
  
  compared with the ideal value of $32$.
- **Parallel efficiency remains very high.**  
  Even at 32 ranks,

  $$
  E(32)=0.986,
  $$

  corresponding to $98.6\%$ efficiency.
- **The scaling trend is stable across repetitions.**
  The standard deviations are small compared with the median runtimes, indicating low run-to-run variability.
- **No clear scaling saturation is observed.**  
  Up to 32 ranks, adding more MPI processes still provides almost proportional performance gains.

#### Factors Supporting Near-Ideal Scaling

The dominant cost is still the direct force calculation.
At 32 ranks, each rank owns

$$
N_{\mathrm{local}}
=
\frac{32768}{32}
=
1024
$$

target particles.

Each of these targets still interacts with all $32768$ source particles, so one rank performs approximately

$$
1024\times32768
\approx
3.36\times10^7
$$

particle interactions per force evaluation.

This means that even at the largest tested process count, each rank still has a large amount of useful computation.

The timing data confirm this: the force kernel still accounts for
$$
97.59\%
$$

of the total runtime at 32 ranks.

Therefore, communication, synchronization, and the remaining integration overheads are still too small to limit scaling significantly.


#### Force-kernel throughput

The throughput measurement confirms the same behaviour from the point of view of the force kernel.

With 100 KDK steps, the solver performs 101 force evaluations: one initial evaluation and one after each step.

The interaction rate is
$$
R_{\mathrm{int}}
=
\frac{N(N-1)\times101}{T_{\mathrm{force}}},
$$

where $T_{\mathrm{force}}$ is the total time spent evaluating forces.

`Ginteraction/s` expresses this rate in billions of ordered particle interactions per second.

| MPI ranks | Force time (s) | Force / total | Ginteraction/s |
|---:|---:|---:|---:|
| 1 | $370.531173 \pm 0.489564$ | 98.97% | 0.293 |
| 2 | $185.020960 \pm 0.084704$ | 98.51% | 0.586 |
| 4 | $92.475055 \pm 0.019084$ | 98.27% | 1.173 |
| 8 | $46.242101 \pm 0.003023$ | 98.08% | 2.345 |
| 16 | $23.142003 \pm 0.001770$ | 97.98% | 4.686 |
| 32 | $11.584479 \pm 0.006580$ | 97.59% | 9.361 |

The interaction throughput increases from

$$
0.293
$$

to

$$
9.361\ \mathrm{Ginteraction/s},
$$

which is almost exactly a $32\times$ increase.

This confirms that *the force kernel itself scales almost linearly with the number of MPI ranks*.

#### Interpretation

The strong-scaling results can therefore be summarized as:

```text
fixed N
→ more MPI ranks
→ less force work per rank
→ runtime decreases almost proportionally

but

force work is still dominant
→ communication/overhead remain small
→ efficiency stays close to 100%
```

$$
\boxed{
\text{the solver shows near-ideal strong scaling up to 32 MPI ranks because the force kernel remains dominant and scales almost linearly}
}
$$

A clear scaling limit is not reached in the tested range. Such a limit would become visible only when the local computation per rank becomes small enough for communication and synchronization overheads to represent a significant fraction of the total runtime.


### 7.4 Weak Scaling

Weak scaling evaluates how performance changes when the problem size grows together with the number of parallel resources.

In this experiment, each MPI rank always owns

$$
N_{\mathrm{local}}=8192
$$

target particles, while the global problem size grows as

$$
N=P\,N_{\mathrm{local}},
$$

where $P$ is the number of MPI ranks.

For a direct all-pairs N-body algorithm, keeping $N_{\mathrm{local}}$ fixed does not imply constant work per rank. Each local target still interacts with all global source particles.

Therefore,

$$
W_{\mathrm{rank}}
\propto
N_{\mathrm{local}}N
=
P\,N_{\mathrm{local}}^2.
$$

The work per rank grows linearly with $P$, so the algorithm-aware ideal runtime is

$$
T_{\mathrm{ideal}}(P)=P\,T(1).
$$

Weak-scaling efficiency is therefore defined as

$$
E_{\mathrm{weak}}(P)
=
\frac{P\,T(1)}{T(P)},
$$

where $T(P)$ is the measured runtime. A value close to 1 means that the application follows the expected direct-N-body scaling.

The scaled speedup shown in the figure is

$$
S_{\mathrm{scaled}}(P)
=
P\,E_{\mathrm{weak}}(P),
$$

with ideal value $P$.

![Native weak scaling](report/figures/weak_scaling_native_final.svg)

| MPI ranks | Global $N$ | Total time (s) | Scaled speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 8,192 | $24.032469 \pm 0.022871$ | 1.000 | 1.000 |
| 2 | 16,384 | $48.688509 \pm 0.049339$ | 1.974 | 0.987 |
| 4 | 32,768 | $97.615508 \pm 0.098695$ | 3.939 | 0.985 |
| 8 | 65,536 | $195.622070 \pm 0.250037$ | 7.862 | 0.983 |
| 16 | 131,072 | $391.787201 \pm 1.201007$ | 15.703 | 0.981 |

#### Main observations
- **Runtime follows the expected linear growth very closely.**  
  $$
  24.0
  \rightarrow
  48.7
  \rightarrow
  97.6
  \rightarrow
  195.6
  \rightarrow
  391.8\ \mathrm{s}.
  $$
  Doubling the number of ranks also doubles the global problem size and approximately doubles the work performed by each rank.
- **Scaled speedup remains close to ideal.**  
  At 16 ranks,
  $$
  S_{\mathrm{scaled}}(16)=15.703
  $$
  compared with the ideal value of $16$.
- **Weak-scaling efficiency remains above $98\%$.**  
  $$
  E_{\mathrm{weak}}(16)=0.981.
  $$
- **The measurements are stable.**  
  The standard deviations remain small compared with the median runtimes.
- **Communication remains a secondary cost.**  
  Its fraction stays at only a few percent, reaching approximately $3.1\%$ at 8 ranks and $2.6\%$ at 16 ranks. The small non-monotonic variation does not affect the overall scaling trend.
- **Numerical correctness is preserved.**  
  The largest measured energy drift is
  $$
  3.58\times10^{-6},
  $$
  well below the validation threshold of $10^{-4}$.

#### Interpretation

The weak-scaling behaviour can be summarized as:

```text
fixed number of targets per rank
        ↓
more MPI ranks
        ↓
larger global N
        ↓
more source particles for every target
        ↓
work per rank grows as P
        ↓
ideal runtime also grows as P
```

The measured runtime follows this expected behaviour very closely. The small decrease in efficiency from $1.000$ to $0.981$ shows that communication and synchronization introduce only limited additional overhead.

$$
\boxed{
\text{the measured weak scaling closely follows the direct-N-body ideal, with efficiency remaining above }98\%
}
$$

The key point is therefore that constant runtime is not the correct weak-scaling expectation for a direct $O(N^2)$ solver. With $N_{\mathrm{local}}$ fixed, the amount of work per rank still increases linearly with the number of MPI ranks.


### 7.5 Scalability and Bottleneck Evolution

The scaling results can be interpreted through Amdahl's and Gustafson's perspectives.

#### Amdahl: fixed global problem
Amdahl's law is relevant to strong scaling, where $N$ is fixed.

$$
S_{\mathrm{Amdahl}}(P)
=
\frac{1}{f+(1-f)/P},
$$

where $f$ represents serial and non-scalable work.
In the measured results:
- ideal speedup at 32 ranks: $32$;
- measured speedup: $31.538$;
- efficiency: $0.986$;
- force fraction decreases only from $98.97\%$ to $97.59\%$.

The small loss of efficiency shows that communication, synchronization, and other overheads are becoming more visible as the work per rank decreases.

However, the force kernel still dominates, so **Amdahl saturation is not yet reached.**

$$
\boxed{
\text{Amdahl appears as the small efficiency loss at high rank counts}
}
$$

#### Gustafson: growing problem size
Gustafson's perspective is relevant to weak scaling, where the problem size increases with the number of ranks:

$$
N=P\,N_{\mathrm{local}}.
$$

For direct N-body, the work per rank is not constant:

$$
W_{\mathrm{rank}}
\propto
P\,N_{\mathrm{local}}^2.
$$

Therefore, the appropriate ideal reference is
$$
T_{\mathrm{ideal}}(P)=P\,T(1).
$$

At 16 ranks:
- ideal scaled speedup: $16$;
- measured scaled speedup: $15.703$;
- weak efficiency: $0.981$.

The additional computational work therefore keeps the extra MPI ranks effectively utilized.

$$
\boxed{
\text{Gustafson appears in the ability to scale the problem while keeping efficiency above }98\%
}
$$

#### Bottleneck evolution
The overall trend is:

```text
fixed N + more ranks
→ less computation per rank
→ overhead becomes relatively more important
→ Amdahl effects increase

growing N + more ranks
→ more useful computation
→ resources remain well utilized
→ Gustafson behaviour
```

Within the tested range, the application remains compute-bound: MPI communication becomes more visible with increasing $P$, but the direct force kernel is still the dominant cost.

## 8. Containerization and Portability

This section evaluates whether the N-body solver can run inside a **Singularity container** with negligible performance overhead while preserving MPI compatibility.

For an MPI application, container portability depends not only on packaging the executable and its dependencies, but also on correct interaction with the **host communication stack**. Container overhead may therefore come from several sources:
- application execution;
- MPI library and transport configuration;
- container startup;
- communication latency and bandwidth.

The analysis follows a top-down approach:

```text
Native vs container application performance
                ↓
Container startup overhead
                ↓
MPI latency and bandwidth
```

The goal is to determine whether any difference observed at application level is caused by the container runtime itself or by the underlying MPI and communication environment.


### 8.1 Container Design, Host MPI Binding and Transport Configuration

The container provides a reproducible userspace while reusing the MPI stack installed on the HPC system at runtime.

#### Container design
The image is based on `Ubuntu 24.04`. Ubuntu 22.04 was initially considered, but it was not compatible with the OpenMPI installation available on Orfeo because the host MPI stack required a newer `glibc`.

A general-purpose Ubuntu image was preferred to a vendor HPC image because the application is CPU-only and does not require GPU libraries, vendor math libraries, or a pre-packaged HPC software stack.
Two versioned recipes are provided:

```text
container/Dockerfile
container/nbody.def
```

The final Singularity image used on Orfeo is

```text
container/nbody_latest.sif
```

and is generated from the definition file with

```text
singularity build container/nbody_latest.sif container/nbody.def
```

The `.sif` file is not committed because it is a generated binary artifact; the reproducible source is the definition file.
All experiments used **SingularityCE 4.3.1**, which was the container runtime available on Orfeo.


#### Host MPI binding

OpenMPI is installed inside the image because `mpicc` is required during the build stage.

However, the production MPI runs deliberately use the OpenMPI installation provided by Orfeo:

```text
build time   → container MPI
runtime      → host MPI
```

This distinction is important because the MPI implementation inside a container may not match the launcher, transport components, or communication stack of the HPC system.

The runtime binding was verified with `ldd` in three environments:

```text
native executable
        ↓
Orfeo MPI libraries

container without host binding
        ↓
container MPI libraries

container with host MPI binding
        ↓
Orfeo MPI libraries
```

In the final configuration,

```text
libmpi
libopen-rte
libopen-pal
```

are resolved from
```text
/opt/programs/openMPI/4.1.6/lib
```

matching the native executable.

Therefore, the final container experiments use the **same cluster MPI implementation as the native runs**, rather than silently falling back to the MPI packaged inside the image.


#### Controlled MPI transport

Initial Singularity tests using the default OpenMPI transport selection produced container-specific warnings.

Two relevant cases were observed:
```text
UCX
→ library version mismatch warning

vader shared memory
→ /dev/shm segment warning
```

Although the application still completed correctly, these effects would make a native-versus-container comparison ambiguous: a performance difference could come from the MPI transport configuration rather than from Singularity itself.

For the final controlled comparison, both native and container runs therefore use

```text
OMPI_MCA_pml=^ucx
OMPI_MCA_btl=self,tcp
OMPI_MCA_osc=^ucx
OMPI_MCA_btl_vader_single_copy_mechanism=none
```

which forces the same clean transport path:
```text
native ─────┐
            ├── self + TCP
container ──┘
```

The efficient `vader` shared-memory transport is therefore disabled for this specific comparison.

This makes absolute intra-node MPI performance more conservative, but *improves the validity* of the comparison because both executions use the same communication path.

After making the MPI binding and transport policy explicit, the final smoke tests and native-versus-container runs completed correctly without the previous UCX or `vader` warnings.

$$
\boxed{
\text{Container overhead can be isolated only after controlling MPI binding and transport}
}
$$


### 8.2 Native vs Container Application Performance

The application-level container cost is evaluated by comparing the complete N-body solver in **native** and **Singularity** execution.

For each configuration, both deployments use the same:
- physical problem and integration parameters;
- number of MPI ranks;
- repetition count;
- host OpenMPI runtime and transport policy defined in Section 8.1.

The measured container overhead is defined as

$$
O_{\mathrm{container}}
=
100
\left(
\frac{T_{\mathrm{container}}}
     {T_{\mathrm{native}}}
-1
\right).
$$

Therefore:

$$
O_{\mathrm{container}}>0
\Rightarrow
\text{container slower},
$$

$$
O_{\mathrm{container}}<0
\Rightarrow
\text{container faster in the measured run}.
$$

However, the two binaries use different compiler targets:
```text
native:     -march=native
container:  -march=x86-64-v3
```

The measured percentage must therefore be interpreted as a **deployment-level difference**, not as the isolated cost of Singularity alone.


#### Strong-scaling comparison

![Strong scaling native vs container](report/figures/strong_scaling_native_container_final.svg)


| MPI ranks | Native time (s) | Container time (s) | Overhead |
|---:|---:|---:|---:|
| 1 | $377.476774 \pm 1.575675$ | $373.803578 \pm 0.225742$ | $-0.97\%$ |
| 2 | $192.136294 \pm 0.051788$ | $187.832592 \pm 0.087729$ | $-2.24\%$ |
| 4 | $96.453229 \pm 0.070562$ | $94.165248 \pm 0.044843$ | $-2.37\%$ |
| 8 | $48.498850 \pm 0.009672$ | $47.260008 \pm 0.013726$ | $-2.55\%$ |
| 16 | $24.492627 \pm 0.008468$ | $23.766715 \pm 0.018591$ | $-2.96\%$ |
| 32 | $12.529375 \pm 0.044831$ | $12.164797 \pm 0.007646$ | $-2.91\%$ |

Main observations
- Native and container curves follow almost the same strong-scaling trend.
- The measured difference remains small across all configurations:
$$
-3.0\% \lesssim O_{\mathrm{container}} \lesssim -1.0\%.
$$
- No positive container penalty is observed up to 32 MPI ranks.
- The scaling behaviour itself is preserved: container execution does not introduce an increasing loss of efficiency as the number of ranks grows.
The negative values must not be interpreted as a Singularity speedup. They only indicate that, in these measurements, the complete container deployment produced slightly lower runtimes.


#### Weak-scaling comparison

![Weak scaling native vs container](report/figures/weak_scaling_native_container_final.svg)

| MPI ranks | Native time (s) | Container time (s) | Overhead |
|---:|---:|---:|---:|
| 1 | $23.332344 \pm 0.036565$ | $23.350336 \pm 0.001905$ | $+0.08\%$ |
| 2 | $46.981598 \pm 0.055411$ | $46.961641 \pm 0.002709$ | $-0.04\%$ |
| 4 | $94.373261 \pm 0.032592$ | $94.132828 \pm 0.026092$ | $-0.25\%$ |
| 8 | $190.051070 \pm 0.027471$ | $188.543249 \pm 0.047304$ | $-0.79\%$ |
| 16 | $386.723009 \pm 1.094676$ | $377.915699 \pm 0.162304$ | $-2.28\%$ |

Main observations
- Native and container weak-scaling curves remain closely aligned.
- The measured overhead ranges from approximately
$$
+0.1\% \text{ to } -2.3\%.
$$
- At low and intermediate rank counts, the difference is very close to zero.
- Even as both the number of ranks and the global problem size increase, no growing positive container penalty appears.

#### Interpretation

The two experiments give the same overall picture:

```text
same MPI runtime and transport
            ↓
native and container
follow almost identical scaling trends
            ↓
runtime differences remain within a few percent
            ↓
no systematic container penalty is observed
```

he small native-versus-container differences should be interpreted carefully:
- they reflect the **complete deployment configuration**, not Singularity alone;
- native and container binaries use different compiler targets;
- normal run-to-run and job-to-job variability may also contribute;
- the reported standard deviation measures only repeatability within the same configuration.
Therefore, negative overhead values do not demonstrate that the container is faster.

The supported conclusion is:
$$
\boxed{
\text{containerization preserves the scaling behaviour and introduces no systematic performance penalty in the tested range}
}
$$

### 8.3 Container Launch Overhead

Container startup cost was measured separately from application execution using a trivial command:

```text
singularity exec <image> true
```

The same experiment was repeated with native execution of `true`. Each configuration was executed ten times.

| Mode | Repeats | Median (s) | Stdev (s) | Min (s) | Max (s) |
|---|---:|---:|---:|---:|---:|
| Container | 10 | 0.102052 | 0.196828 | 0.100407 | 0.724725 |
| Native | 10 | 0.000444 | 0.000120 | 0.000432 | 0.000823 |

The typical Singularity launch cost is therefore approximately

$$
T_{\mathrm{launch}}\approx0.102\ \mathrm{s},
$$

corresponding to a fixed overhead relative to native execution of

$$
0.102052-0.000444
\approx
0.1016\ \mathrm{s}.
$$

#### Main observations
- **Container startup introduces a fixed cost of about $0.1$ s.**
- Native execution of the trivial command is essentially instantaneous in comparison.
- One container launch reached approximately
  $$
  0.725\ \mathrm{s},
  $$ 
  which explains the relatively *large sample standard deviation*.
- No measurement was removed. Because the distribution contains this unusually slow launch, the **median** is more representative of the typical startup cost than the mean would be.


#### Interpretation

The launch overhead is primarily a **fixed cost**:

```text
start container
      ↓
≈ 0.1 s startup cost
      ↓
run application
```

Its importance therefore depends on application runtime.

For very short commands, $0.1$ s can be significant. For the N-body experiments, however, execution times range from several seconds to several minutes, so the same fixed cost is strongly amortized.

Even compared with the shortest application runs of approximately $12$ s, a $0.1$ s launch cost represents less than $1\%$ of the total runtime.

Therefore, container startup cannot explain a large performance difference in the application-level scaling experiments.

$$
\boxed{
\text{Singularity startup costs about }0.1\text{ s, but this fixed overhead becomes negligible for long HPC runs}
}
$$


### 8.4 MPI Communication Microbenchmark

Possible MPI-specific container overhead was isolated using the OSU Micro-Benchmarks, measuring point-to-point latency and bandwidth.

OSU was built in user space with the same OpenMPI environment used by the application. Both native and container tests used:
- 2 MPI ranks;
- host MPI binding;
- the same controlled `self,tcp` transport defined in Section 8.1;
- five repetitions per configuration.

Because `self,tcp` is deliberately used instead of the native shared-memory `vader` path, these measurements are **not intended to represent the maximum intra-node performance of Orfeo**. Their purpose is to compare native and container MPI communication under identical conditions.

![OSU MPI microbenchmark: native vs container](report/figures/mpi_microbenchmark_curve.svg)

Representative results are:

| Metric | Message size | Native | Container | Difference |
|:---|---:|---:|---:|---:|
| Latency | 1 B | $9.940 \pm 0.161\ \mu s$ | $10.000 \pm 0.088\ \mu s$ | $+0.60\%$ |
| Bandwidth | 4 MiB | $1673.450 \pm 245.560$ MB/s | $1668.570 \pm 244.222$ MB/s | $-0.29\%$ |


#### Main observations
- **The native and container curves are almost superimposed** over the tested message sizes.
- For a 1-byte message, container latency increases by only
  $$
  10.000-9.940=0.060\ \mu s,
  $$
  corresponding to
  $$
  +0.60\%.
  $$
- For a 4 MiB message, bandwidth changes by only
  $$
  -0.29\%.
  $$
- The bandwidth difference is much smaller than the observed run-to-run variability, so it does not indicate a meaningful communication penalty.

#### Interpretation

The controlled experiment shows:

```text
same host MPI
+ same transport
        ↓
native and container
show nearly identical
latency and bandwidth
        ↓
no systematic MPI penalty
from containerization
```

The result does **not** show that `self,tcp` is the optimal transport for Orfeo. It shows that, once MPI binding and transport are controlled, Singularity does not introduce a measurable systematic degradation in MPI communication.

This is consistent with the application-level results: communication remains a secondary cost compared with the direct force computation.

$$
\boxed{
\text{With identical MPI binding and transport, native and container communication performance is essentially unchanged}
}
$$

### 8.5 Portability Considerations
The container experiments show that *application portability can be achieved without materially changing performance*, provided that the **MPI runtime is integrated explicitly**.

The final setup relies on three principles:
- **portable userspace**: the container provides a reproducible software environment without assuming the exact host architecture;
- **host MPI binding**: the cluster OpenMPI installation is injected at runtime and verified with `ldd`;
- **controlled transport**: native and container comparisons use the same MPI transport policy.

This separates the portable part of the application from the cluster-specific communication layer:

```text
portable
source code + container userspace + MPI algorithm
                    ↓
cluster-dependent
MPI libraries + transport + interconnect configuration
```

The MPI ring algorithm itself uses standard point-to-point operations and therefore does not depend directly on vendor-specific APIs. However, the MPI implementation may internally rely on shared-memory mechanisms or network-specific components that must remain available inside the container.

For this reason, deployment on another HPC system requires revalidating:
- the host MPI implementation and library binding;
- the available transport components;
- the target interconnect;
- basic MPI correctness and communication performance.

For example, a multi-node InfiniBand system would keep the same application code and ring decomposition, but would require an MPI transport configuration appropriate for that fabric. The controlled `self,tcp` policy used here for native-versus-container comparison should therefore not be interpreted as a production configuration for a different cluster.

$$
\boxed{
\text{The application is portable, while MPI integration remains cluster-dependent}
}
$$


## 9. Limitations and Reproducibility

The benchmark campaign follows the objectives of the assignment while adapting them to the resources available on Orfeo.

The main limitations are:
- **Platform**: experiments were performed on the Orfeo GENOA system rather than LEONARDO.
- **Single-node execution**: all final MPI scaling and container comparisons were limited to one compute node.
- **Problem size**: strong- and weak-scaling sizes were reduced relative to the assignment reference values because of the two-hour wall-time limit.
- **Hardware counters**: perf, PAPI, and LIKWID counters were not available. Vectorization and kernel behaviour were therefore evaluated through GCC optimization reports, timing, and derived throughput.
- **Container comparison**: native and container binaries use different compilation targets: native: `-march=native` and container: `-march=x86-64-v3`

Therefore, measured native-versus-container differences represent the complete deployment configurations rather than the isolated cost of Singularity.

- **MPI transport**: native-only performance experiments use the host default MPI transport, whereas native-versus-container experiments use the controlled transport defined in Section 8.1. This difference is intentional because the two experiments answer different questions.

These limitations restrict how far the results can be generalized, but do not affect the internal comparison of configurations performed under the same experimental conditions.


## 10. Conclusions

This project evaluated a direct N-body solver from four complementary perspectives: **numerical correctness, kernel optimization, parallel scalability, and containerized execution**.

#### Main findings
- **Numerical correctness**  
  The KDK integration with softened gravity remained within the selected energy-drift tolerance in all final validation experiments.
- **Kernel optimization**  
  Reducing the theoretical operation count does not automatically improve performance. Newton's third law introduces update dependencies, while reciprocal-square-root approximations provide high isolated throughput but no solver-level benefit without effective vectorization.
  The most useful low-level optimization was accumulator splitting, reaching up to
  $$
  1.151\times
  $$
  speedup by increasing instruction-level parallelism.
- **Parallel scalability**
  The direct force kernel remains the dominant cost throughout the tested range.
  Strong scaling reaches
  $$
  E(32)=0.986,
  $$
  while algorithm-aware weak scaling reaches
  $$
  E_{\mathrm{weak}}(16)=0.981.
  $$
  Communication and synchronization become progressively more visible, but no clear scaling saturation is reached.
- **Communication overlap**
  Non-blocking MPI exposes the possibility of overlapping communication and computation, but provides little total-runtime improvement because communication is still a small fraction of execution time.
- **Hybrid MPI/OpenMP mapping**
  The tested mappings perform similarly on one GENOA node. Hardware topology provides useful guidance, but the best decomposition must ultimately be determined experimentally.
- **Containerization**
  Singularity preserves the application's scaling behaviour when host MPI binding and transport are controlled explicitly. Application-level differences remain within a few percent, and the OSU microbenchmarks show no systematic MPI communication penalty.

#### Overall interpretation
The experiments show a consistent performance picture:

 ```text
direct O(N²) force kernel
        ↓
dominant computational cost
        ↓
good parallel scalability
        ↓
MPI overhead remains secondary
        ↓
container overhead is strongly amortized
```

The main current limitation is therefore not MPI or containerization, but the lack of effective vectorization in the dominant interaction loop.

#### Future work
A natural next step is to combine:

```text
SoA layout
    +
explicit SIMD
    +
rsqrt refinement
    +
multiple accumulators
```

into a single vectorized force kernel.
This would directly target the dominant computational bottleneck and could allow the operation-level advantages observed in the microbenchmarks to translate into solver-level speedup.
$$
\boxed{
\text{Performance is ultimately determined by how well optimizations match the structure of the dominant force kernel}
}
$$

## Appendix A

### A.1 Repository Structure and Reproducibility

The project repository is organized so that the numerical implementation, benchmark execution, post-processing, container recipe, and report artifacts remain separated. The main files used for the final results are:

```text
Nbody_serial/
├── nbody_common.h
│   └── shared particle type, precision type, math helpers, CLI utilities
│
├── nbody_core.h, nbody_core.c
│   └── force kernels, energy diagnostics, Leapfrog/KDK integration helpers
│
├── nbody_direct_serial.c
│   └── serial OpenMP-capable baseline for validation and kernel studies
│
├── nbody_mpi_omp.c
│   └── MPI + OpenMP production solver with ring-shift communication
│
├── generate_ic.c
│   └── Plummer initial-condition generator used by validation and benchmarks
│
├── benchmark_layout.c
├── benchmark_rsqrt_kernel.c
│   └── focused microbenchmarks for data layout and SIMD reciprocal square root
│
├── Makefile
│   └── native/MPI builds, precision selection, OpenMP flags
│
├── container/
│   ├── Dockerfile
│   │   └── Docker build environment required by the assignment
│   └── nbody.def
│       └── Singularity definition file for the Orfeo .sif image
│
├── scripts/
│   ├── benchmark/
│   │   └── benchmark drivers for validation, scaling, hybrid mapping,
│   │       and kernel trade-off experiments
│   ├── slurm/
│   │   └── SLURM entry points used on Orfeo
│   ├── analyze/
│   │   └── CSV summarization, table generation, and SVG plotting
│   ├── smoke/
│   │   └── native, MPI, and container smoke tests
│   └── utils/
│       └── system information, MPI binding checks, launch-overhead measurement
│
├── report/
│   ├── data/
│   │   └── hardware and software stack snapshot
│   ├── tables/
│   │   └── final CSV and Markdown tables used in the report
│   └── figures/
│       └── final SVG plots used in the report
│
└── FINAL_REPORT.md
    └── final project report
```

This layout mirrors the experimental structure of the project: source files implement the solver, `scripts/benchmark` and `scripts/slurm` produce the raw measurements, `scripts/analyze` converts them into report-ready artifacts, and `report/` stores only the final tables, figures, and setup information used in the discussion.


### A.2 Benchmark Traceability

All report tables and figures are generated from benchmark data and analysis scripts rather than manually edited results.

| Assignment item | Report section | Reproducible artifact |
|:---|:---|:---|
| Energy validation | Section 2.4 | `report/tables/validation_energy_summary.md` |
| $O(N^2)$ growth | Section 5.2 | `report/tables/n_growth_summary.md` |
| Kernel profiling and throughput | Sections 5.3, 7.3 | `report/tables/force_throughput_summary.md` |
| Newton, layout, rsqrt, accumulators | Section 6 | `report/tables/*_tradeoff_summary.md` |
| Hybrid mapping and binding | Sections 4.4, 7.1 | `scripts/slurm/binding_check.slurm`, `report/tables/hybrid_mapping_summary.md` |
| Communication overlap | Section 7.2 | `report/tables/ring_overlap_summary.md` |
| Strong/weak scaling | Sections 7.3, 7.4 | `report/tables/*scaling*_summary.md` |
| Container launch and MPI tests | Sections 8.3, 8.4 | `report/tables/container_launch_overhead_summary.md`, `report/tables/mpi_microbenchmark_summary.md` |
| Native/container comparison | Section 8.2 | `report/tables/container_overhead_summary.md` |

Derived quantities such as `Ginteraction/s`, speedup, efficiency, and container overhead are generated by the corresponding scripts under `scripts/analyze/`.