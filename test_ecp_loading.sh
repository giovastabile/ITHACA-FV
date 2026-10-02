#!/bin/sh
# Compatibility entry point for the active ECP rule check.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$repo/test_ecp_weights_updated.sh" "$@"
