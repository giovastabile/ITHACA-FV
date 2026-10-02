#!/bin/sh
# Build and run the ECP/full-ROM comparison. Source OpenFOAM and ITHACA first.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$repo/tutorials/CFD/12simpleSteadyNS_ECP/Allrun" "$@"
