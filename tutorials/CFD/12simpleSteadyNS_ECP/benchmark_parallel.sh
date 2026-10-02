#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ranks=${1:-4}
results_root=${2:-"$case_dir/benchmark-results"}

case "$ranks" in
    ''|*[!0-9]*)
        echo "Usage: $0 [ranks>=2] [results-directory]" >&2
        exit 2
        ;;
esac
if [ "$ranks" -lt 2 ]; then
    echo "Usage: $0 [ranks>=2] [results-directory]" >&2
    exit 2
fi

: "${WM_PROJECT_VERSION:?Source the OpenFOAM environment first}"
: "${LIB_ITHACA:?Source ITHACA-FV/etc/bashrc first}"
: "${FOAM_USER_APPBIN:?Source the OpenFOAM environment first}"

if [ "$WM_PROJECT_VERSION" != "v2606" ]; then
    echo "This benchmark case requires OpenFOAM v2606 (found $WM_PROJECT_VERSION)" >&2
    exit 2
fi

app="$FOAM_USER_APPBIN/12simpleSteadyNS_ECP"
for command_name in "$app" "$case_dir/prepare_parallel.py" \
    "$(command -v decomposePar || true)" "$(command -v mpirun || true)" \
    "$(command -v python3 || true)" /usr/bin/time
do
    if [ -z "$command_name" ] || [ ! -x "$command_name" ]; then
        echo "Required executable not found: ${command_name:-unknown}" >&2
        echo "Build the tutorial first with ./Allrun" >&2
        exit 2
    fi
done

if [ ! -d "$case_dir/ITHACAoutput/Offline" ]; then
    echo "Missing offline snapshots: $case_dir/ITHACAoutput/Offline" >&2
    exit 2
fi

active_rule="$case_dir/ITHACAoutput/12simpleSteadyNS_ECP/active"
parallel_basis="$case_dir/ITHACAoutput/12simpleSteadyNS_ECP/parallelBasis"
if [ ! -f "$active_rule/metadata" ] || [ ! -d "$parallel_basis/0" ]; then
    echo "Missing prepared ECP rule or parallel basis in ITHACAoutput." >&2
    echo "Prepare them first with ./Allrun_parallel, then retry." >&2
    exit 2
fi

ecp_nodes=$(awk '$1 == "ecpNodes" { gsub(/;/, "", $2); print $2; exit }' \
    "$case_dir/system/ITHACAdict")
if [ "$ecp_nodes" != "100" ]; then
    echo "Expected ecpNodes 100 in $case_dir/system/ITHACAdict (found ${ecp_nodes:-unset})" >&2
    exit 2
fi

cache_folder=$(cat "$active_rule/cache.txt")
case "$cache_folder" in
    ITHACAoutput/12simpleSteadyNS_ECP/ECP_projected_v1_*) ;;
    *)
        echo "Unexpected active ECP cache reference: $cache_folder" >&2
        exit 2
        ;;
esac
if [ ! -d "$case_dir/$cache_folder" ]; then
    echo "Missing active ECP cache: $case_dir/$cache_folder" >&2
    exit 2
fi

mkdir -p "$results_root"
results_root=$(CDPATH= cd -- "$results_root" && pwd)
timestamp=$(date '+%Y%m%d-%H%M%S')
results_dir="$results_root/comparison-$timestamp"
work_dir="$results_dir/work"
mkdir -p "$work_dir"

cleanup()
{
    result=$?

    if [ "$result" -eq 0 ]; then
        rm -rf "$work_dir"
    else
        echo "Benchmark failed; temporary cases and logs are preserved in $results_dir" >&2
    fi

    exit "$result"
}
trap cleanup EXIT HUP INT TERM

serial_seed="$work_dir/serial-seed"
mkdir -p "$serial_seed"
cp -a "$case_dir/0" "$case_dir/constant" "$case_dir/system" \
    "$case_dir/lift" "$case_dir/par" "$case_dir/vel.txt" "$serial_seed/"
mkdir -p "$serial_seed/ITHACAoutput"
cp -a "$case_dir/ITHACAoutput/Offline" "$serial_seed/ITHACAoutput/"
mkdir -p "$serial_seed/ITHACAoutput/12simpleSteadyNS_ECP"
cp -a "$active_rule" "$serial_seed/ITHACAoutput/12simpleSteadyNS_ECP/"
cp -a "$parallel_basis" "$serial_seed/ITHACAoutput/12simpleSteadyNS_ECP/"
mkdir -p "$serial_seed/$(dirname "$cache_folder")"
cp -a "$case_dir/$cache_folder" "$serial_seed/$cache_folder"
cp "$case_dir/prepare_parallel.py" "$serial_seed/"

# Keep diagnostic and full-ROM validation work out of the timing comparison.
sed -i -E \
    -e 's/^[[:space:]]*debug[[:space:]]+.*/debug false;/' \
    -e 's/^[[:space:]]*hrEquivalenceTests[[:space:]]+.*/hrEquivalenceTests false;/' \
    -e 's/^[[:space:]]*ecpValidate[[:space:]]+.*/ecpValidate false;/' \
    "$serial_seed/system/ITHACAdict"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

expected_solves=$(awk 'NR == 1 { print NF; exit }' "$case_dir/par")
if [ -z "$expected_solves" ] || [ "$expected_solves" -lt 1 ]; then
    echo "Could not determine parameter count from $case_dir/par" >&2
    exit 2
fi

repeat=1
while [ "$repeat" -le 2 ]; do
    serial_case="$work_dir/serial-$repeat"
    mkdir -p "$serial_case"
    cp -a "$serial_seed/." "$serial_case/"

    printf 'Running serial benchmark (repeat %s/2)...\n' "$repeat"
    (
        cd "$serial_case"
        /usr/bin/time -f 'WALL_SECONDS=%e' \
            -o "$results_dir/serial-$repeat.wall" \
            "$app" > "$results_dir/serial-$repeat.log" 2>&1
    )

    parallel_case="$work_dir/parallel-$repeat"
    python3 "$serial_seed/prepare_parallel.py" \
        --case "$serial_case" \
        --ranks "$ranks" \
        --output "$parallel_case" \
        > "$results_dir/parallel-$repeat.case"
    if ! cmp -s \
        "$serial_case/ITHACAoutput/12simpleSteadyNS_ECP/active/nodePoints.npy" \
        "$parallel_case/constant/ecpRule/nodePoints.npy" \
        || ! cmp -s \
        "$serial_case/ITHACAoutput/12simpleSteadyNS_ECP/active/quadratureWeights.npy" \
        "$parallel_case/constant/ecpRule/quadratureWeights.npy"; then
        echo "Serial and parallel cases do not share the same ECP rule" >&2
        exit 1
    fi

    printf 'Decomposing parallel case (repeat %s/2)...\n' "$repeat"
    (
        cd "$parallel_case"
        decomposePar -time 0 > "$results_dir/decompose-$repeat.log" 2>&1
    )

    printf 'Running %s-rank benchmark (repeat %s/2)...\n' "$ranks" "$repeat"
    (
        cd "$parallel_case"
        /usr/bin/time -f 'WALL_SECONDS=%e' \
            -o "$results_dir/parallel-$repeat.wall" \
            mpirun -np "$ranks" "$app" -parallel \
            > "$results_dir/parallel-$repeat.log" 2>&1
    )

    repeat=$((repeat + 1))
done

python3 - "$results_dir" "$ranks" "$expected_solves" "$WM_PROJECT_VERSION" \
    "$case_dir" "$ecp_nodes" <<'PY'
import re
import sys
from pathlib import Path

results = Path(sys.argv[1])
ranks = int(sys.argv[2])
expected = int(sys.argv[3])
version = sys.argv[4]
case_dir = Path(sys.argv[5])
ecp_nodes = int(sys.argv[6])


def solver_rows(path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("ROM_TIMING method=HR "):
            continue
        values = dict(
            (key, float(value))
            for key, value in re.findall(
                r"(setup_s|solve_s|reconstruct_s|export_s)=([0-9.eE+-]+)",
                line,
            )
        )
        match = re.search(r"mu=([0-9.eE+-]+) iterations=(\d+)", line)
        if match and "solve_s" in values:
            rows.append(
                {
                    "mu": float(match.group(1)),
                    "iterations": int(match.group(2)),
                    **values,
                }
            )
    return rows


def wall_seconds(path):
    match = re.search(r"WALL_SECONDS=([0-9.eE+-]+)", path.read_text())
    if not match:
        raise SystemExit(f"Could not read wall time from {path}")
    return float(match.group(1))


all_rows = {}
all_walls = {}
for repeat in (1, 2):
    serial_log = results / f"serial-{repeat}.log"
    parallel_log = results / f"parallel-{repeat}.log"
    serial = solver_rows(serial_log)
    parallel = solver_rows(parallel_log)
    if len(serial) != expected or len(parallel) != expected:
        raise SystemExit(
            f"Expected {expected} timed solves per run; got "
            f"{len(serial)} serial and {len(parallel)} parallel in repeat {repeat}"
        )
    if [row["mu"] for row in serial] != [row["mu"] for row in parallel]:
        raise SystemExit(f"Serial and parallel parameter sequences differ in repeat {repeat}")
    for log in (serial_log, parallel_log):
        if "ECP test PASSED" not in log.read_text(errors="replace"):
            raise SystemExit(f"Benchmark did not pass: {log}")

    all_rows[f"serial-{repeat}"] = serial
    all_rows[f"parallel-{repeat}"] = parallel
    all_walls[f"serial-{repeat}"] = wall_seconds(
        results / f"serial-{repeat}.wall"
    )
    all_walls[f"parallel-{repeat}"] = wall_seconds(
        results / f"parallel-{repeat}.wall"
    )

serial_solve = sum(
    row["solve_s"]
    for repeat in (1, 2)
    for row in all_rows[f"serial-{repeat}"]
)
serial_iterations = sum(
    row["iterations"]
    for repeat in (1, 2)
    for row in all_rows[f"serial-{repeat}"]
)
parallel_iterations = sum(
    row["iterations"]
    for repeat in (1, 2)
    for row in all_rows[f"parallel-{repeat}"]
)
parallel_solve = sum(
    row["solve_s"]
    for repeat in (1, 2)
    for row in all_rows[f"parallel-{repeat}"]
)
serial_wall = sum(all_walls[f"serial-{repeat}"] for repeat in (1, 2)) / 2
parallel_wall = sum(
    all_walls[f"parallel-{repeat}"] for repeat in (1, 2)
) / 2
solve_speedup = serial_solve / parallel_solve
iteration_speedup = (
    serial_solve / serial_iterations
) / (
    parallel_solve / parallel_iterations
)
iterations = [row["iterations"] for row in all_rows["serial-1"]]
parallel_iterations_first = [
    row["iterations"] for row in all_rows["parallel-1"]
]

report = f"""Tutorial 12 ECP: 1-core vs {ranks}-rank online comparison
==========================================================
Source case: {case_dir}
OpenFOAM: {version}
ECP rule: {ecp_nodes} global cells; same prepared rule used by serial and parallel runs
Parameters per run: {expected}
Repeats: 2
Iteration counts per parameter in first repeat:
  1 core : {", ".join(map(str, iterations))}
  {ranks} ranks: {", ".join(map(str, parallel_iterations_first))}
Solver convergence: ECP test passed in all four runs; full-ROM accuracy
validation, debug, and equivalence diagnostics were disabled in the isolated
benchmark copies only.

Mean whole-process wall time per run:
  1 core : {serial_wall:.3f} s
  {ranks} ranks: {parallel_wall:.3f} s
(These include different serial and parallel initialization work; not a direct
speedup metric.)

Sum of solver-reported SIMPLE loop time ({2 * expected} solves):
  1 core : {serial_solve:.3f} s ({serial_solve / (2 * expected):.3f} s/solve)
  {ranks} ranks: {parallel_solve:.3f} s ({parallel_solve / (2 * expected):.3f} s/solve)
  online solve-loop speedup: {solve_speedup:.2f}x
  mean time per SIMPLE iteration: {serial_solve / serial_iterations * 1000:.3f} ms serial,
    {parallel_solve / parallel_iterations * 1000:.3f} ms parallel ({iteration_speedup:.2f}x)

The same ECP rule and exported basis from each serial run were used in its
paired parallel run. Parallel mesh decomposition is excluded from the
online solve-loop speedup.

Raw logs and wall-time records are in this directory. Temporary case copies
were removed after the successful comparison.
"""
(results / "summary.txt").write_text(report)
print(report)
PY

printf 'Results saved in %s\n' "$results_dir"
