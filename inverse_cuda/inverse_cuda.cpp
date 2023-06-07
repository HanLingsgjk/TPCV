#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <iostream>

#include "inverse.cuh"

int inverse_forward_cuda(at::Tensor& pixel, at::Tensor& corr, at::Tensor& flow)
{

  int batch = pixel.size(0);
  int height = pixel.size(1);
  int weight = pixel.size(2);


  int success = inverse_forward_cuda_kernel(pixel,corr, flow, batch, height, width, at::cuda::getCurrentCUDAStream())
  );

  //check for errors
  if (!success) {
    AT_ERROR("CUDA call failed");
  }

  return 1;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &inverse_forward_cuda, "Correlation forward (CUDA)");
}
