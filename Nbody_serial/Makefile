CC        ?= cc
MPICC     ?= mpicc
STD       ?= -std=c11
CPPFLAGS  ?=
CFLAGS    ?= -O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic
LDLIBS    ?= -lm
PRECISION ?= double
OPENMP    ?= 0

PROGRAMS := nbody_direct_serial generate_ic inspect_particles benchmark_layout
MPI_PROGRAMS := nbody_mpi_omp
EXTRA_PROGRAMS := benchmark_rsqrt_kernel
HEADERS  := nbody_common.h nbody_core.h
BUILD_DIR := build
COMMON_FLAGS_FILE := $(BUILD_DIR)/common.flags
SOLVER_FLAGS_FILE := $(BUILD_DIR)/solver.flags

ifeq ($(PRECISION),float)
PRECISION_CPPFLAGS := -DNBODY_USE_FLOAT
else ifeq ($(PRECISION),double)
PRECISION_CPPFLAGS := -DNBODY_USE_DOUBLE
else
$(error PRECISION must be either double or float)
endif

ifeq ($(OPENMP),1)
OPENMP_CFLAGS ?= -fopenmp
OPENMP_LDLIBS ?= -fopenmp
else ifneq ($(OPENMP),0)
$(error OPENMP must be either 0 or 1)
endif

.PHONY: all mpi clean run-smoke FORCE

all: $(PROGRAMS)

mpi: $(MPI_PROGRAMS)

nbody_direct_serial: nbody_direct_serial.c nbody_core.c $(HEADERS) $(COMMON_FLAGS_FILE) $(SOLVER_FLAGS_FILE)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OPENMP_CFLAGS) -o $@ nbody_direct_serial.c nbody_core.c $(LDLIBS) $(OPENMP_LDLIBS)

generate_ic: generate_ic.c $(HEADERS) $(COMMON_FLAGS_FILE)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

inspect_particles: inspect_particles.c $(HEADERS) $(COMMON_FLAGS_FILE)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

benchmark_layout: benchmark_layout.c nbody_core.c $(HEADERS) $(COMMON_FLAGS_FILE) $(SOLVER_FLAGS_FILE)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OPENMP_CFLAGS) -o $@ benchmark_layout.c nbody_core.c $(LDLIBS) $(OPENMP_LDLIBS)

benchmark_rsqrt_kernel: benchmark_rsqrt_kernel.c nbody_core.c $(HEADERS) $(COMMON_FLAGS_FILE)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ benchmark_rsqrt_kernel.c nbody_core.c $(LDLIBS)

nbody_mpi_omp: nbody_mpi_omp.c nbody_core.c $(HEADERS) $(COMMON_FLAGS_FILE) $(SOLVER_FLAGS_FILE)
	$(MPICC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OPENMP_CFLAGS) -o $@ nbody_mpi_omp.c nbody_core.c $(LDLIBS) $(OPENMP_LDLIBS)

$(BUILD_DIR):
	mkdir -p $@

$(COMMON_FLAGS_FILE): FORCE | $(BUILD_DIR)
	@printf '%s\n' '$(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) $(LDLIBS)' > $@.tmp; \
	if test -f $@ && cmp -s $@ $@.tmp; then rm $@.tmp; else mv $@.tmp $@; fi

$(SOLVER_FLAGS_FILE): FORCE | $(BUILD_DIR)
	@printf '%s\n' '$(OPENMP_CFLAGS) $(OPENMP_LDLIBS)' > $@.tmp; \
	if test -f $@ && cmp -s $@ $@.tmp; then rm $@.tmp; else mv $@.tmp $@; fi

FORCE:

run-smoke: all
	./generate_ic --model 0 --n 128 --seed 42 --output plummer_128.bin
	./nbody_direct_serial --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output plummer_128_final.bin --quiet
	./generate_ic --model 1 --n 128 --seed 43 --output ball_128.bin
	./nbody_direct_serial --input ball_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output ball_128_final.bin --quiet

clean:
	rm -rf $(PROGRAMS) $(MPI_PROGRAMS) $(EXTRA_PROGRAMS) *.o *.bin $(BUILD_DIR)
