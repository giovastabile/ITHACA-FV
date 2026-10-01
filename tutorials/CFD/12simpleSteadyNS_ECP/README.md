# Tutorial 12: Empirical Cubature

This standalone variant adds empirical cubature (ECP) to the steady SIMPLE ROM in Tutorial 12. The original residual-sampling tutorial is unchanged.

## Workflow

The offline phase computes the full-order snapshots, lift, and POD bases. ECP uses cellwise velocity and pressure basis/snapshot features to select cubature cells and weights. `ecpLayers` adds a stencil halo around selected cells; only the selected cells contribute to weighted reduced projections. `ecpNodes` controls the requested cubature point count.

The online SIMPLE solve applies the same ECP weights to the momentum projection, pressure projection, and pressure-gradient contribution. The ECP implementation currently requires a serial mesh.

The training features in this starter version are state-space fields, not the complete nonlinear momentum and pressure integrand snapshots. Treat accuracy as problem-dependent and compare reconstructed fields and reduced residuals with the full-mesh ROM before relying on the reduction for production predictions. A stronger training set should include operator-contribution snapshots from representative SIMPLE iterations.

## Build and run

From this directory, build with `wmake` and run `12simpleSteadyNS_ECP` in the case directory. Adjust `ecpNodes` and `ecpLayers` in `system/ITHACAdict` for the mesh and desired accuracy.
