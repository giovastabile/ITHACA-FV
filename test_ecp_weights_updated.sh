#!/bin/sh
# Check the active rule exported by the tutorial, not an arbitrary old cache.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case_dir=${1:-"$repo/tutorials/CFD/12simpleSteadyNS_ECP"}
exec python3 - "$case_dir" <<'PY'
from pathlib import Path
import re
import sys
import numpy as np

case = Path(sys.argv[1])
rule = case / "ITHACAoutput/12simpleSteadyNS_ECP/active"
try:
    weights = np.load(rule / "quadratureWeights.npy", allow_pickle=False).reshape(-1)
    nodes = np.load(rule / "nodePoints.npy", allow_pickle=False).reshape(-1)
    volumes = np.load(rule / "cellVolumes.npy", allow_pickle=False).reshape(-1)
    if weights.size == 0 or weights.size != nodes.size:
        raise ValueError("weights and node counts must match and be nonempty")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and nonzero")
    if not np.issubdtype(nodes.dtype, np.integer):
        raise ValueError("cell indices must be integers")
    if np.unique(nodes).size != nodes.size or (nodes < 0).any() or (nodes >= volumes.size).any():
        raise ValueError("cell indices must be unique and inside the mesh")
    if not np.isfinite(volumes).all() or (volumes <= 0).any():
        raise ValueError("cell volumes must be finite and positive")
    volume_error = abs(weights @ volumes[nodes] - volumes.sum()) / volumes.sum()
    if volume_error > 1e-4:
        raise ValueError(f"cubature volume error is too large: {volume_error:.3g}")

    mask_text = (case / "0/ecpMask").read_text()
    nonuniform = re.search(r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;", mask_text, re.S)
    uniform = re.search(r"internalField\s+uniform\s+([\d.eE+-]+)\s*;", mask_text)
    if nonuniform:
        mask = np.fromstring(nonuniform.group(2), sep=" ")
        if int(nonuniform.group(1)) != mask.size:
            raise ValueError("invalid mask field length")
    elif uniform:
        mask = np.full(volumes.size, float(uniform.group(1)))
    else:
        raise ValueError("cannot read the ASCII ecpMask internal field")
    if mask.size != volumes.size or not np.array_equal(np.flatnonzero(mask == 1), np.sort(nodes)):
        raise ValueError("ecpMask selected cells do not match the active cubature rule")
except (OSError, ValueError) as error:
    sys.exit(f"ECP weights FAILED: {error}\nRun test_ecp_implementation.sh to generate the active rule.")
print(f"ECP weights PASSED: {nodes.size}/{volumes.size} cells; volume error {volume_error:.3g}")
print("Field accuracy and convergence are checked by test_ecp_implementation.sh.")
PY
