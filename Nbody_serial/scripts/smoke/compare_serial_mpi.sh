#!/bin/sh
set -eu

N=${N:-128}
NSTEPS=${NSTEPS:-5}
DT=${DT:-1e-4}
EPS=${EPS:-0.05}
MASS=${MASS:-1.0}
SEED=${SEED:-123}
ENERGY_EVERY=${ENERGY_EVERY:-$NSTEPS}
SERIAL_THREADS=${SERIAL_THREADS:-1}
OMP_THREADS=${OMP_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
RANKS=${RANKS:-"1 2 4"}
OUTPUT_DIR=${OUTPUT_DIR:-results}
INPUT=${INPUT:-$OUTPUT_DIR/plummer_${N}.bin}
CSV=${CSV:-$OUTPUT_DIR/serial_mpi_compare_$(date +%Y%m%d_%H%M%S).csv}

mkdir -p "$OUTPUT_DIR"

make OPENMP=1 PRECISION=double
make mpi OPENMP=1 PRECISION=double

if [ ! -f "$INPUT" ]; then
  ./generate_ic \
    --model 0 \
    --n "$N" \
    --seed "$SEED" \
    --scale 1.0 \
    --mass "$MASS" \
    --output "$INPUT"
fi

printf '%s\n' \
  "backend,n,nsteps,dt,eps,mass,ranks,threads,total_seconds,force_seconds,communication_seconds,energy_seconds,final_total_energy,max_relative_energy_drift,status" \
  > "$CSV"

extract_final_field () {
  key=$1
  awk -v key="$key" '
    $1 == "#" && $2 == "final:" {
      for (i = 1; i <= NF; ++i) {
        split($i, field, "=")
        if (field[1] == key) {
          print field[2]
          found = 1
          exit
        }
      }
    }
    END {
      if (!found)
        print "nan"
    }
  '
}

extract_timing () {
  key=$1
  awk -v key="$key" '
    $1 == "#" && $2 == "timing" && $3 == key {
      print $4
      found = 1
      exit
    }
    END {
      if (!found)
        print "0"
    }
  '
}

extract_serial_energy () {
  awk -v step="$NSTEPS" '
    $1 == step {
      print $5
      found = 1
    }
    END {
      if (!found)
        print "nan"
    }
  '
}

extract_mpi_energy () {
  awk -v step="$NSTEPS" '
    $1 == step {
      print $3
      found = 1
    }
    END {
      if (!found)
        print "nan"
    }
  '
}

append_row () {
  backend=$1
  ranks=$2
  threads=$3
  total_seconds=$4
  force_seconds=$5
  communication_seconds=$6
  energy_seconds=$7
  final_total_energy=$8
  drift=$9
  status=${10}

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$backend" \
    "$N" \
    "$NSTEPS" \
    "$DT" \
    "$EPS" \
    "$MASS" \
    "$ranks" \
    "$threads" \
    "$total_seconds" \
    "$force_seconds" \
    "$communication_seconds" \
    "$energy_seconds" \
    "$final_total_energy" \
    "$drift" \
    "$status" \
    >> "$CSV"
}

run_mpi () {
  ranks=$1

  if [ $((N % ranks)) -ne 0 ]; then
    echo "error: N=$N must be divisible by ranks=$ranks" >&2
    exit 1
  fi

  if command -v srun >/dev/null 2>&1; then
    srun -n "$ranks" -c "$OMP_THREADS" \
      ./nbody_mpi_omp \
        --input "$INPUT" \
        --nsteps "$NSTEPS" \
        --dt "$DT" \
        --eps "$EPS" \
        --mass "$MASS" \
        --energy-every "$ENERGY_EVERY" \
        --timing
  elif command -v mpirun >/dev/null 2>&1; then
    mpirun -np "$ranks" \
      ./nbody_mpi_omp \
        --input "$INPUT" \
        --nsteps "$NSTEPS" \
        --dt "$DT" \
        --eps "$EPS" \
        --mass "$MASS" \
        --energy-every "$ENERGY_EVERY" \
        --timing
  else
    echo "error: neither srun nor mpirun was found" >&2
    exit 1
  fi
}

echo "# serial vs MPI comparison"
echo "# input=$INPUT"
echo "# n=$N nsteps=$NSTEPS dt=$DT eps=$EPS mass=$MASS"
echo "# csv=$CSV"

serial_output=$(
  OMP_NUM_THREADS="$SERIAL_THREADS" \
  OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
  OMP_PLACES="${OMP_PLACES:-cores}" \
  ./nbody_direct_serial \
    --input "$INPUT" \
    --nsteps "$NSTEPS" \
    --dt "$DT" \
    --eps "$EPS" \
    --mass "$MASS" \
    --energy-every "$ENERGY_EVERY" \
    --integrator kdk \
    --timing
)

printf '%s\n' "$serial_output"

append_row \
  "serial_kdk" \
  "1" \
  "$SERIAL_THREADS" \
  "$(printf '%s\n' "$serial_output" | extract_timing total_seconds)" \
  "$(printf '%s\n' "$serial_output" | extract_timing force_seconds)" \
  "0" \
  "$(printf '%s\n' "$serial_output" | extract_timing energy_seconds)" \
  "$(printf '%s\n' "$serial_output" | extract_serial_energy)" \
  "$(printf '%s\n' "$serial_output" | extract_final_field max_relative_energy_drift)" \
  "$(printf '%s\n' "$serial_output" | extract_final_field status)"

for ranks in $RANKS; do
  echo "# MPI run ranks=$ranks omp_threads=$OMP_THREADS"
  mpi_output=$(
    OMP_NUM_THREADS="$OMP_THREADS" \
    OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
    OMP_PLACES="${OMP_PLACES:-cores}" \
    run_mpi "$ranks"
  )

  printf '%s\n' "$mpi_output"

  append_row \
    "mpi_kdk" \
    "$ranks" \
    "$OMP_THREADS" \
    "$(printf '%s\n' "$mpi_output" | extract_timing total_seconds)" \
    "$(printf '%s\n' "$mpi_output" | extract_timing force_seconds)" \
    "$(printf '%s\n' "$mpi_output" | extract_timing communication_seconds)" \
    "$(printf '%s\n' "$mpi_output" | extract_timing energy_seconds)" \
    "$(printf '%s\n' "$mpi_output" | extract_mpi_energy)" \
    "$(printf '%s\n' "$mpi_output" | extract_final_field max_relative_energy_drift)" \
    "$(printf '%s\n' "$mpi_output" | extract_final_field status)"
done

echo "# wrote $CSV"

