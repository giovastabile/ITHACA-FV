# Tutorial 12: Empirical Cubature

This serial tutorial runs an empirical-cubature (ECP) steady SIMPLE ROM and,
by default, checks its reconstructed fields against the full-mesh ROM.

## Build and run

Source your OpenFOAM environment and `ITHACA-FV/etc/bashrc`, then run from this
directory:

```sh
./Allrun > log.ecp 2>&1
```

`Allrun` rebuilds `ITHACA_CORE`, `ITHACA_ROMPROBLEMS`, and the tutorial. A successful run ends
with `ECP test PASSED` and exits with status zero. Nonconvergence, non-finite
comparison errors, or field errors above `ecpValidationTolerance` produce a
nonzero exit status. Each parameter prints its relative volume-weighted L2
velocity and pressure errors.

Results are written to `ITHACAoutput/ReconstructECP` and the reference results
to `ITHACAoutput/ReconstructFullROM`. `0/ecpMask` shows the selected cells and
assembly halo. Existing offline snapshots and POD modes are reused; regenerate
those outputs if changing the offline parameter set or POD configuration.

## Comparing serial and parallel performance

With the prepared ECP rule and basis available in `ITHACAoutput`, run:

```sh
./benchmark_parallel.sh
```

The script performs two serial/four-rank online comparisons using the same
100-cell ECP rule, disables validation and diagnostics only in disposable
benchmark copies, and writes a summary and raw logs under
`benchmark-results/`. An optional rank count and results directory can be
provided, for example `./benchmark_parallel.sh 4 /tmp/ecp-results`.
`./Allclean` removes the default `benchmark-results/` directory and preserves
the offline snapshots and ECP caches.

## Training and settings

ECP fits **cell contributions to the reduced operators**: each entry of the
relaxed momentum and pressure matrices, their right-hand sides, and the
pressure-gradient projection. Training uses the same lift/POD basis ordering
as the online solver, includes finite-volume boundary contributions, and
normalizes the feature columns and compresses their span before fitting
nonnegative weights. Matrix entries
already include cell-volume integration; gradient features include it
explicitly. The resulting weights multiply cell sums, rather than replacing
cell volumes.

The settings in `system/ITHACAdict` are:

- `NmodesUproj`, `NmodesPproj`: 5 each for this test. The velocity dimension
  includes the lift and four POD modes, matching the full ROM's convention.
- `ecpNodes`: maximum cubature cells (300 by default, out of 9000). The fit
  stops earlier when `ecpTolerance` is reached; this case selects 218 cells.
- `ecpLayers`: face-neighbour halo for equation assembly (8 by default).
- `ecpTrainingSnapshots`: evenly spaced snapshots per parameter (2 by default),
  in addition to the lift-only online initial state. Cached snapshots must be
  grouped evenly by parameter.
- `ecpBasisTolerance`: relative singular-value cutoff for training compression
  (1e-6). This case retains 218 independent training directions.
- `ecpTolerance`: relative fitting tolerance for the compressed basis (1e-8).
- `ecpValidate`: run the full-mesh ROM comparison (true by default). Disable
  for sampled-only runs after validating the chosen settings.
- `ecpValidationTolerance`: maximum relative L2 error for **each** field (0.05).
- `hrEquivalenceTests`: print first-iteration operator diagnostics. The
  pre-relaxation sampling/uniform-weight tests are unweighted diagnostics;
  the post-relaxation, gradient, and pressure tests use the ECP weights.

The quadrature cache is keyed by the requested point count, fitting tolerance,
and a fingerprint of the compressed training matrix. Changes to training data invalidate it.

The supplied five-parameter test converges for both solvers. With 218 cubature
cells and an eight-layer halo (6531 assembly cells), relative errors against
the full ROM were at most 0.014% for velocity and 0.34% for pressure on
OpenFOAM v2606. Setting `ecpNodes` to the mesh cell count uses the exact
unit-weight rule for a full/sampled equivalence check.

A small point budget or a four-layer halo is insufficient for this case.
Repeated SIMPLE corrections require a larger halo than the first-iteration
stencil alone. Increase the
cubature count or enrich the training states if the comparison fails for a
new case. Convergence of the sampled equations alone does not establish
accuracy. The comparison checks agreement with the full ROM, not FOM accuracy
or speedup; the halo can be much larger than the selected-cell set.

## Checking the exported rule

From the repository root, `./test_ecp_implementation.sh` runs the same build and
accuracy test as `Allrun`. After it finishes, `./test_ecp_weights_updated.sh`
checks finite nonnegative weights, unique cell indices, volume conservation,
and agreement with `0/ecpMask`. The other root weight/loading scripts delegate
to this check. It deliberately fails if the active outputs are missing or
invalid; checking files alone does not establish field accuracy.

The current rule is exported to
`ITHACAoutput/12simpleSteadyNS_ECP/active/{quadratureWeights,nodePoints,cellVolumes}.npy`.
`active/cache.txt` identifies its cache. There is no
`ITHACAoutput/Offline/cubatureWeights.tx` file.
