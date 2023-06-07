#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <iostream>

#include "inverse.cuh"

at::Tensor inverse_forward_cuda(at::Tensor& pixel, at::Tensor& corr, at::Tensor& flow, at::Tensor& pv)
{

  int batch = pixel.size(0);
  int height = pixel.size(1);
  int weight = pixel.size(2);
  at::Tensor output;

  output = inverse_forward_cuda_kernel(pixel,corr, flow, pv, batch, height, weight, at::cuda::getCurrentCUDAStream());
  
  return output;

}
at::Tensor inverse_nearset_cuda(at::Tensor& pixel, at::Tensor& corr, at::Tensor& flow, at::Tensor& pv)
{

  int batch = pixel.size(0);
  int height = pixel.size(1);
  int weight = pixel.size(2);
  at::Tensor output;

  output = inverse_nearest_cuda_kernel(pixel,corr, flow, pv, batch, height, weight, at::cuda::getCurrentCUDAStream());
  
  return output;

}
at::Tensor inverse_index_cuda(at::Tensor& output, at::Tensor& index, at::Tensor& iout)
{

  int batch = output.size(0);
  int height = output.size(1);
  int weight = output.size(2);
  at::Tensor output1;
  output1 = inverse_index_cuda_kernel(output, index, iout, batch, height, weight, at::cuda::getCurrentCUDAStream());
  
  return output1;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &inverse_forward_cuda, "Correlation forward (CUDA)");
  m.def("nearset", &inverse_nearset_cuda, "Correlation nearset (CUDA)");
  m.def("index", &inverse_index_cuda, "Correlation index (CUDA)");
}
