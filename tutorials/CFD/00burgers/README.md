# Instructions

## Compute the modes

~~~bash
cp -r 0.original 0
blockMesh && checkMesh && setExprFields
00burgers train
cd ITHACAoutput/red_coeff/
python3 red_coeff_mat.py
~~~

## Utilities

To plot the eigenvalues:

```
python3 plot_evals.py
```

To plot the cumulative eigenvalues:

```
python3 plot_cumulative.py
```

To clip the negative values of the snapshots to zero:

```bash
python3 cut_snapshots_below_zero.py
```

## POD-Galerkin Non-intrusive LSTM

~~~bash
cd NonIntrusive
python3 non_intrusive_lstm.py
python3 predict.py
~~~

## POD-Galerkin Intrusive

~~~bash
00burgers test
~~~

## Dimension reduction with Convolutional Autoencoder  and Non-intrusive model

~~~bash
cd Autoencoders/ConvolutionalAe/
python3 train.py
python3 train_lstm.py
python3 predict_lstm.py
python3 compute_error.py
~~~

## Nonlinear Manifold LSPG

~~~bash
00burgers nonlinear
~~~

## Plot errors

~~~bash
python3 plot_errors.py
~~~

## Data

nonIntrusiveCoeff.npy = coefficients of the modes for nonintrusive POD-Galerkin

npInitialAndModes.npy = initial and selected modes for projection of POD-Galerkin

npSnapshots.npy = training snapshots from full order model

npSnapshots_cut.npy = training snapshots with negative values clipped to zero

npTrueSnapshots.npy = test snapshots from full order model

parTest.npy = parameters of test set

parTrain.npy = parameters of training set

snapshotsConvAeTrueProjection.npy = snapshots reconstructed with the autoencoder (passed through the encoder and decoder). Needed for projection error

snapshotsReconstructedConvAe.npy = snapshots reconstructed from predicted latent variables from LSTM net. Needed for nonIntrusiveConvAe error

# TODO

1. **FIX**: 00burgers nonlinear does not update solution when the true Jacobian from the decoder is employed (ok with Eigen numDiff and options Forward and Central)
2. Dimension reduction with variational autoencoders
3. Dimension reduction with shallow net as decoder

