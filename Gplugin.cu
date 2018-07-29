#include "GpluginGPU.h"

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
    
__global__ void PReLU(const int n, const int channels, const int dim,
    const float* in, float* out, const float* slope_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}


cudaError_t PReLUForward(const int count, const int channels, const int dim, const float* bottom_data,
  float* top_data, void* mDeviceKernel, const int div_factor){
  PReLU<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,channels,dim,bottom_data,top_data,static_cast<const float*>(mDeviceKernel),div_factor);
  return cudaPeekAtLastError();
}
