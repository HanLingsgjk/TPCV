#include <torch/torch.h>
#include <iostream>
#include <torch/extension.h>
//corr : b,h,w,h,w  I1->I2
//pixel: b,h,w,4 (xl,xr,yu,yd)
//flow : b,2,h,w

torch::Tensor inverse_opticalflow(torch::Tensor pixel,torch::Tensor corr,torch::Tensor flow) {
  // 
  int b = pixel.size(0);
  int h = pixel.size(1);
  int w = pixel.size(2);
  
  float lurd_pv[b][h][w][4][3];
  int bl = h*w*4*3;
  int hl = w*4*3;
  
  int bp = h*w*4;
  int hp = w*4;
  
  int bc = h*w*h*w;
  int I1h = w*h*w;
  int I1w = h*w;
  
  int bf = h*w*2;
  int tf = h*w;
  for (int ba=0;ba<b;ba++)
  	for (int i=0;i<h;i++)
  	      for (int j=0;j<w;j++)
  	           for (int n=0;n<4;n++)
  	           {
  	           	lurd_pv[ba][i][j][n][0] = 0;
  	           	lurd_pv[ba][i][j][n][1] = 0;
  	           	lurd_pv[ba][i][j][n][2] = -10000;
  	           }
  for (int ba=0;ba<b;ba++)
  	for (int i=0;i<h;i++)
  	      for (int j=0;j<w;j++)
  	      {
  	      	int xl = pixel.data_ptr<float>()[bp*ba+hp*i+j*4];
  	      	int xr = pixel.data_ptr<float>()[bp*ba+hp*i+j*4+1];
  	      	int yu = pixel.data_ptr<float>()[bp*ba+hp*i+j*4+2];
  	      	int yd = pixel.data_ptr<float>()[bp*ba+hp*i+j*4+3];
  	      	if( (xl>=0) & (xr<w) & (yd<h) & (yu>=0))
  	      	{
  	      		if (corr.data_ptr<float>()[bc*ba+I1h*i+I1w*j+w*yu+xl]> lurd_pv[ba][yu][xl][0][2])
  	      		{
  	      			lurd_pv[ba][yu][xl][0][0] = -flow.data_ptr<float>()[bf*ba+0*tf+w*i+j];
  	      			lurd_pv[ba][yu][xl][0][1] = -flow.data_ptr<float>()[bf*ba+1*tf+w*i+j];
  	      			lurd_pv[ba][yu][xl][0][2] = corr.data_ptr<float>()[bc*ba+I1h*i+I1w*j+w*yu+xl];
  	      		}
  	      		
  	      		if (corr.data_ptr<float>()[bc*ba+I1h*i+I1w*j+w*yu+xr]> lurd_pv[ba][yu][xr][1][2])
  	      		{
  	      			lurd_pv[ba][yu][xr][1][0] = -flow.data_ptr<float>()[bf*ba+0*tf+w*i+j];
  	      			lurd_pv[ba][yu][xr][1][1] = -flow.data_ptr<float>()[bf*ba+1*tf+w*i+j];
  	      			lurd_pv[ba][yu][xr][1][2] = corr.data_ptr<float>()[bc*ba+I1h*i+I1w*j+w*yu+xr];
  	      		}
  	      		
  	      		if (corr.data_ptr<float>()[bc*ba+I1h*i+I1w*j+w*yd+xr]> lurd_pv[ba][yd][xr][2][2])
  	      		{
  	      			lurd_pv[ba][yd][xr][2][0] = -flow.data_ptr<float>()[bf*ba+0*tf+w*i+j];
  	      			lurd_pv[ba][yd][xr][2][1] = -flow.data_ptr<float>()[bf*ba+1*tf+w*i+j];
  	      			lurd_pv[ba][yd][xr][2][2] = corr.data_ptr<float>()[bc*ba+I1h*i+I1w*j+w*yd+xr];
  	      		}
  	      		
  	      		if (corr.data_ptr<float>()[bc*ba+I1h*i+I1w*j+w*yd+xl]> lurd_pv[ba][yd][xl][3][2])
  	      		{
  	      			lurd_pv[ba][yd][xl][3][0] = -flow.data_ptr<float>()[bf*ba+0*tf+w*i+j];
  	      			lurd_pv[ba][yd][xl][3][1] = -flow.data_ptr<float>()[bf*ba+1*tf+w*i+j];
  	      			lurd_pv[ba][yd][xl][3][2] = corr.data_ptr<float>()[bc*ba+I1h*i+I1w*j+w*yd+xl];
  	      		}
  	      			

  	      	}
  	      }
                 
  torch::Tensor output = torch::from_blob((float *)&lurd_pv, /*sizes=*/{b,h,w,4,3});
  return output.clone();
  // END output_tensor
}
TORCH_LIBRARY(my_ops, m) {
  m.def("inverse_opticalflow", inverse_opticalflow);
}
