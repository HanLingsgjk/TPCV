#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <cuda_runtime.h>

at::Tensor inverse_forward_cuda_kernel(at::Tensor& pixel, at::Tensor& corr, at::Tensor& flow, at::Tensor& lurd_pv, int batch, int height, int width, cudaStream_t stream);
at::Tensor inverse_nearest_cuda_kernel(at::Tensor& pixel, at::Tensor& corr, at::Tensor& flow, at::Tensor& lurd_pv, int batch, int height, int width, cudaStream_t stream);
at::Tensor inverse_index_cuda_kernel(at::Tensor& output, at::Tensor& index, at::Tensor& iout, int batch, int height, int width, cudaStream_t stream);
