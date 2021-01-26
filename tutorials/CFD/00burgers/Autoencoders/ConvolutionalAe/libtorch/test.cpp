#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include "cnpy.H"

int main() {

  cnpy::load(initial_latent, "./Autoencoders/ConvolutionalAe/latent_initial_4.npy");

  model = autoPtr<torch::jit::script::Module>(new torch::jit::script::Module(torch::jit::load("../decoder_gpu_4.pt")));

  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;

}