# Instructions

## Compute the modes

~~~bash
cp -r 0.original 0
blockMesh && checkMesh && setExprFields
00burgers train
cd ITHACAoutput/red_coeff/
python3 red_coeff_mat.py
~~~

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



# TODO

1. **FIX**: 00burgers nonlinear does not update solution when the true Jacobian from the decoder is employed (ok with Eigen numDiff and options Forward and Central)
2. Dimension reduction with variational autoencoders
3. Dimension reduction with shallow net as decoder

