from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from glob import glob
import os
cu_srcs = glob(os.path.join("ABQ", "csrc", "mma_any", "**","*.cu"), recursive=True)

# TODO: auto detect GPU ARCH
nvcc_args = [
    '-gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\"'
]
cxx_args = ['-std=c++14']

setup(
    name='ABQ',
    version='0.1.0',
    description='Arbitrary Bit Quantization Gemm.',
    install_requires=['numpy', 'torch'],
    packages=['ABQ'],
    ext_modules=[
        CUDAExtension(name='ABQ_cuda', 
                      sources=['ABQ/csrc/ABQ_cuda.cu',] + cu_srcs,
                      extra_compile_args={
                          'cxx': cxx_args,
                          'nvcc': nvcc_args
                          },
                      define_macros=[('GPU_ARCH',80)]), 
    ],
    include_dirs=['ABQ/csrc', "../3rdparty/cutlass/include", "../3rdparty/cutlass/tools/util/include"],
    cmdclass={'build_ext': BuildExtension},
)