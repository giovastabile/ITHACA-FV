## Instructions

cp -r 0.original 0

blockMesh && checkMesh

setExprFields

00burgers train

python3 ITHACAoutput/red_coeff/red_coeff_mat.py

python3 non_intrusive.py

00burgers test

python3 Autoencoders/ConvolutionalAe/train.py

python3 Autoencoders/ConvolutionalAe/predict.py

python3 Autoencoders/ConvolutionalAe/compute_error.py

python3 plot_error.py