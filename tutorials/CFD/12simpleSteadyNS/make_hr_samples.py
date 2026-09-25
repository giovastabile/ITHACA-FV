#!/usr/bin/env python3
import re
import sys
from pathlib import Path

fraction = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
if not (0.0 < fraction <= 1.0):
    raise SystemExit("fraction must be in (0,1]")

candidates = [Path("0/C"), Path("0.0/C")]
Cfile = next((p for p in candidates if p.exists()), None)

if Cfile is None:
    raise SystemExit(
        "Cannot find 0/C. Run first:\n"
        "    postProcess -func writeCellCentres -time 0"
    )

text = Cfile.read_text()

m = re.search(
    r'internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\(',
    text,
    flags=re.MULTILINE,
)

if not m:
    raise SystemExit(f"Cannot determine number of cells from {Cfile}")

n_cells = int(m.group(1))
n_samples = max(1, round(fraction * n_cells))

labels = sorted({
    min(n_cells - 1, int((i + 0.5) * n_cells / n_samples))
    for i in range(n_samples)
})

out = Path("system/topoSetDict")
out.parent.mkdir(parents=True, exist_ok=True)

body = "\n".join(f"            {i}" for i in labels)

content = f'''FoamFile
{{
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}}

actions
(
    {{
        name    hrSamples;
        type    cellSet;
        action  new;
        source  labelToCell;

        value
        (
{body}
        );
    }}
);
'''

out.write_text(content)

print(f"Mesh cells      : {n_cells}")
print(f"Requested frac. : {fraction:.4f}")
print(f"Sample cells    : {len(labels)} ({100*len(labels)/n_cells:.2f}%)")
print(f"Wrote           : {out}")
