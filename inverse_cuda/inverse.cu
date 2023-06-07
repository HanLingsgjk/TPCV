#include <stdio.h>
#include "inverse.cuh"
#include <torch/torch.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

//corr : b,h,w,h,w  I1->I2
//corr_dis : b,h,w,4

//pixel: b,h,w,4,2 (xl,xr,yu,yd)
//flow : b,2,h,w
//output : b,h,w,4,5

//lurd_pv : b,h,w,4,5(ifx,ify,v,i,j)
//index   :b,h,w,1
//iout:    b,h,w,5
using at::Half;
using scalar_t = float;
template <typename scalar_t>
__global__ void inverse_forward(const scalar_t* pixel, const scalar_t* corr, const scalar_t* flow, float *output ,int bl,int hl,int bf,int tf, int bp,int hp,int bc,int I1h,int I1w,int h, int w,int idx,int idy)
{

	// n (batch size), c (num of channels), y (height), x (width)
	int ba = blockIdx.x;
	int i = 2*blockIdx.y+idy;
	int j = 2*threadIdx.x+idx;
	int p = threadIdx.y;
	int x = pixel[bp*ba+hp*i+j*4*2+p*2];
	int y = pixel[bp*ba+hp*i+j*4*2+p*2+1];
	if ((x>=0)&(x<w)&(y<h)&(y>=0))
	{
		if(corr[bc*ba+I1h*i+I1w*j+w*y+x] > output[bl*ba+hl*y+x*4*5+p*5+2])
		{
  			output[bl*ba+hl*y+x*4*5+p*5+0] = -flow[bf*ba+0*tf+w*i+j];
  			output[bl*ba+hl*y+x*4*5+p*5+1] = -flow[bf*ba+1*tf+w*i+j];
  			output[bl*ba+hl*y+x*4*5+p*5+2] = corr[bc*ba+I1h*i+I1w*j+w*y+x];
  			output[bl*ba+hl*i+j*4*5+p*5+3] = x;
  			output[bl*ba+hl*i+j*4*5+p*5+4] = y;
  			
  		}
	}
}

template <typename scalar_t>
__global__ void inverse_nearest_forward(const scalar_t* pixel, const scalar_t* corr_dis, const scalar_t* flow, float *output ,int bl,int hl,int bf,int tf, int bp,int hp,int bc,int hc,int h, int w,int idx,int idy)
{

	// n (batch size), c (num of channels), y (height), x (width)
	int ba = blockIdx.x;
	int i = 2*blockIdx.y+idy;
	int j = 2*threadIdx.x+idx;
	int p = threadIdx.y;
	int x = pixel[bp*ba+hp*i+j*4*2+p*2];
	int y = pixel[bp*ba+hp*i+j*4*2+p*2+1];
	if ((x>=0)&(x<w)&(y<h)&(y>=0))
	{
		if(corr_dis[bc*ba+hc*i+4*j+p] > output[bl*ba+hl*y+x*4*5+p*5+2])
		{
  			output[bl*ba+hl*y+x*4*5+p*5+0] = -flow[bf*ba+0*tf+w*i+j];
  			output[bl*ba+hl*y+x*4*5+p*5+1] = -flow[bf*ba+1*tf+w*i+j];
  			output[bl*ba+hl*y+x*4*5+p*5+2] = corr_dis[bc*ba+hc*i+4*j+p];
  			output[bl*ba+hl*i+j*4*5+p*5+3] = x;
  			output[bl*ba+hl*i+j*4*5+p*5+4] = y;
  			
  		}
	}
}

template <typename scalar_t>
__global__ void inverse_index(const scalar_t* output, const scalar_t* index, float *iout,int bl,int hl,int bi,int hi, int bo,int ho, int h, int w)
{

	// n (batch size), c (num of channels), y (height), x (width)
	int ba = blockIdx.x;
	int i = blockIdx.y;
	int j = threadIdx.x;
	int idx = index[bi*ba+hi*i+j];
	iout[bo*ba+ho*i+j*5] = output[bl*ba+hl*i+j*4*5+idx*5];
	iout[bo*ba+ho*i+j*5+1] = output[bl*ba+hl*i+j*4*5+idx*5+1];
	iout[bo*ba+ho*i+j*5+2] = output[bl*ba+hl*i+j*4*5+idx*5+2];
	iout[bo*ba+ho*i+j*5+3] = output[bl*ba+hl*i+j*4*5+idx*5+3];
	iout[bo*ba+ho*i+j*5+4] = output[bl*ba+hl*i+j*4*5+idx*5+4];
}
at::Tensor inverse_forward_cuda_kernel(at::Tensor& pixel, at::Tensor& corr, at::Tensor& flow, at::Tensor& lurd_pv, int batch, int height, int width, cudaStream_t stream)
{
	dim3 blocks_grid(batch,height/2);
	dim3 threads_block(width/2,4);
	
	int bl = height*width*4*5;
  	int hl = width*4*5;
	
	int bf = height*width*2;
  	int tf = height*width;
  	
	int bp = height*width*4*2;
  	int hp = width*4*2;
  
  	int bc = height*width*height*width;
  	int I1h = width*height*width;
  	int I1w = height*width;
  

  	auto output = lurd_pv.detach();//at::empty({batch, 2, height, width}, pixel.options());
  	
	inverse_forward<scalar_t> <<< blocks_grid, threads_block, 0, stream >>> (pixel.data<scalar_t>(),corr.data<scalar_t>(),flow.data<scalar_t>(),output.data<scalar_t>(),bl,hl,bf,tf,bp,hp,bc,I1h,I1w,height,width,0,0);
	cudaDeviceSynchronize();
	inverse_forward<scalar_t> <<< blocks_grid, threads_block, 0, stream >>> (pixel.data<scalar_t>(),corr.data<scalar_t>(),flow.data<scalar_t>(),output.data<scalar_t>(),bl,hl,bf,tf,bp,hp,bc,I1h,I1w,height,width,0,1);
	cudaDeviceSynchronize();
	inverse_forward<scalar_t> <<< blocks_grid, threads_block, 0, stream >>> (pixel.data<scalar_t>(),corr.data<scalar_t>(),flow.data<scalar_t>(),output.data<scalar_t>(),bl,hl,bf,tf,bp,hp,bc,I1h,I1w,height,width,1,0);
	cudaDeviceSynchronize();
	inverse_forward<scalar_t> <<< blocks_grid, threads_block, 0, stream >>> (pixel.data<scalar_t>(),corr.data<scalar_t>(),flow.data<scalar_t>(),output.data<scalar_t>(),bl,hl,bf,tf,bp,hp,bc,I1h,I1w,height,width,1,1);
	cudaError_t err = cudaGetLastError();


	// check for errors
	if (err != cudaSuccess) {
		printf("error in correlation_forward_cuda_kernel: %s\n", cudaGetErrorString(err));
	}

	return output;
}
at::Tensor inverse_nearest_cuda_kernel(at::Tensor& pixel, at::Tensor& corr, at::Tensor& flow, at::Tensor& lurd_pv, int batch, int height, int width, cudaStream_t stream)
{
	dim3 blocks_grid(batch,height/2);
	dim3 threads_block(width/2,4);
	
	int bl = height*width*4*5;
  	int hl = width*4*5;
	
	int bf = height*width*2;
  	int tf = height*width;
  	
	int bp = height*width*4*2;
  	int hp = width*4*2;
  
  	int bc = height*width*4;
  	int hc = width*4;
  

  	auto output = lurd_pv.detach();//at::empty({batch, 2, height, width}, pixel.options());
  	
	inverse_nearest_forward<scalar_t> <<< blocks_grid, threads_block, 0, stream >>> (pixel.data<scalar_t>(),corr.data<scalar_t>(),flow.data<scalar_t>(),output.data<scalar_t>(),bl,hl,bf,tf,bp,hp,bc,hc,height,width,0,0);
	cudaDeviceSynchronize();
	inverse_nearest_forward<scalar_t> <<< blocks_grid, threads_block, 0, stream >>> (pixel.data<scalar_t>(),corr.data<scalar_t>(),flow.data<scalar_t>(),output.data<scalar_t>(),bl,hl,bf,tf,bp,hp,bc,hc,height,width,0,1);
	cudaDeviceSynchronize();
	inverse_nearest_forward<scalar_t> <<< blocks_grid, threads_block, 0, stream >>> (pixel.data<scalar_t>(),corr.data<scalar_t>(),flow.data<scalar_t>(),output.data<scalar_t>(),bl,hl,bf,tf,bp,hp,bc,hc,height,width,1,0);
	cudaDeviceSynchronize();
	inverse_nearest_forward<scalar_t> <<< blocks_grid, threads_block, 0, stream >>> (pixel.data<scalar_t>(),corr.data<scalar_t>(),flow.data<scalar_t>(),output.data<scalar_t>(),bl,hl,bf,tf,bp,hp,bc,hc,height,width,1,1);
	cudaError_t err = cudaGetLastError();


	// check for errors
	if (err != cudaSuccess) {
		printf("error in correlation_forward_cuda_kernel: %s\n", cudaGetErrorString(err));
	}

	return output;
}
at::Tensor inverse_index_cuda_kernel(at::Tensor& output, at::Tensor& index, at::Tensor& iout, int batch, int height, int width, cudaStream_t stream)
{
	dim3 blocks_grid(batch,height);
	dim3 threads_block(width);
	
	int bl = height*width*4*5;
  	int hl = width*4*5;
	
	int bi = height*width;
  	int hi = width;
  
	int bo = height*width*5;
  	int ho = width*5;
  	
  	auto outputi = iout.detach();
	inverse_index<scalar_t> <<< blocks_grid, threads_block, 0, stream >>> (output.data<scalar_t>(),index.data<scalar_t>(),outputi.data<scalar_t>(),bl,hl,bi,hi,bo,ho,height,width);

	cudaError_t err = cudaGetLastError();


	// check for errors
	if (err != cudaSuccess) {
		printf("error in correlation_forward_cuda_kernel: %s\n", cudaGetErrorString(err));
	}

	return outputi;
}
