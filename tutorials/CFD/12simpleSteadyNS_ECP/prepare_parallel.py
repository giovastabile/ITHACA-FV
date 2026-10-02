#!/usr/bin/env python3
"""Stage a fresh online case from a serial -ecpPrepare export."""
import argparse
from pathlib import Path
import re
import shutil
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--ranks', type=int, required=True)
parser.add_argument('--output', type=Path, help='New directory; must not exist')
args = parser.parse_args()
if args.ranks < 2:
    parser.error('--ranks must be at least 2')
case = args.case.resolve()
export = case / 'ITHACAoutput/12simpleSteadyNS_ECP'
rule = export / 'active'
basis = export / 'parallelBasis/0'
if not (rule / 'metadata').is_file() or not (basis / 'ecpLift0').is_file():
    parser.error('Run 12simpleSteadyNS_ECP -ecpPrepare in the serial case first')
if args.output:
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
else:
    target = Path(tempfile.mkdtemp(prefix=f'ParallelCase-np{args.ranks}-', dir=export))
for name in ('constant', 'system'):
    shutil.copytree(case / name, target / name, symlinks=False)
skip = shutil.ignore_patterns('Paux', 'Uaux', 'ecpMask', 'C', 'Cx', 'Cy', 'Cz', 'phi', 'uniform', 'Ulift*')
shutil.copytree(case / '0', target / '0', symlinks=False, ignore=skip)
for field in basis.iterdir():
    if field.is_file():
        shutil.copy2(field, target / '0' / field.name)
shutil.copytree(rule, target / 'constant/ecpRule')
for name in ('par', 'vel.txt'):
    shutil.copy2(case / name, target / name)
# Retain the requested decomposition method, changing only the rank count.
p = target / 'system/decomposeParDict'
text, count = re.subn(r'numberOfSubdomains\s+\d+\s*;', f'numberOfSubdomains {args.ranks};', p.read_text())
if count != 1:
    parser.error('Expected one numberOfSubdomains entry in decomposeParDict')
p.write_text(text)
print(target)
