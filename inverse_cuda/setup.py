import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_61,code=compute_61',
    '-ccbin', '/usr/bin/gcc'
]


setup(
    name='inverse_cuda',
    ext_modules=[
        CUDAExtension('inverse_cuda', [
            'inverse_cuda.cc',
            'inverse.cu'
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
