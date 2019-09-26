# import the necessary packages
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# initialize the matrix dimensions
M = 4
N = 4

# initialize the matrices
a = np.ones((M, N), dtype = np.float16)
b = np.ones((M, N), dtype = np.float16)

# allocate memory for the matrices on the GPU
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)

# copy the matrices from the host to the GPU
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# define the GPU function in C
module = SourceModule("""
    __global__ void add(float *a, float *b)
    {
        int idx = threadIdx.x + threadIdx.y * blockDim.x;        
        a[idx] = a[idx] + b[idx];
    }
""")

# create a callable variable to execute the function
fx = module.get_function("add")

# execute the function
fx(a_gpu, b_gpu, block = (M, N, 1))

# copy the result from the GPU onto the host
result = np.empty_like(a)
cuda.memcpy_dtoh(result, a_gpu)

print(result)

import multiprocessing as mp
p = mp.Process()

p.