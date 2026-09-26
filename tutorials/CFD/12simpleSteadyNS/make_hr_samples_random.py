#!/usr/bin/env python3

import random
import re
import sys
from pathlib import Path

fraction = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

if not (0.0 < fraction <= 1.0):
    raise SystemExit("fraction must be in (0, 1]")

candidates = [Path("0/C"), Path("0.0/C")]
c_file = next((p for p in candidates if p.exists()), None)

if c_file is None:
    raise SystemExit(
        "Cannot find 0/C.\n"
        "Run first:\n"
        "    postProcess -func writeCellCentres -time 0"
    )

text = c_file.read_text()

match = re.search(
    r'internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\(',
    text,
    flags=re.MULTILINE,
)

if not match:
    raise SystemExit(f"Cannot determine number of cells from {c_file}")

n_cells = int(match.group(1))
n_samples = max(1, round(fraction * n_cells))

rng = random.Random(seed)
labels = sorted(rng.sample(range(n_cells), n_samples))

labels_block = "\n".join(
    f"            {label}"
    for label in labels
)

topo_set_dict = (
    "FoamFile\n"
    "{\n"
    "    format      ascii;\n"
    "    class       dictionary;\n"
    "    object      topoSetDict;\n"
    "}\n\n"
    "actions\n"
    "(\n"
    "    // Random interior sampling\n"
    "    {\n"
    "        name    hrSamples;\n"
    "        type    cellSet;\n"
    "        action  new;\n"
    "        source  labelToCell;\n\n"
    "        value\n"
    "        (\n"
    + labels_block + "\n"
    "        );\n"
    "    }\n\n"
    "    // Add cells adjacent to physical boundaries\n"
    "    {\n"
    "        name    hrSamples;\n"
    "        type    cellSet;\n"
    "        action  add;\n"
    "        source  patchToCell;\n\n"
    "        patches\n"
    "        (\n"
    "            inlet\n"
    "            outlet\n"
    "            walls\n"
    "        );\n"
    "    }\n"
    ");\n"
)

out = Path("system/topoSetDict")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(topo_set_dict)

print(f"Mesh cells                  : {n_cells}")
print(f"Requested interior fraction : {fraction:.4f}")
print(f"Random seed                 : {seed}")
print(f"Interior sampled cells      : {len(labels)}")
print("Boundary patches added      : inlet outlet walls")
print("Excluded patch              : frontAndBack")
print(f"Wrote                       : {out}")
