#include "oneflow/customized/kernels/random_mask_generator.h"

namespace oneflow {

namespace {

__global__ void SetupKernel(int64_t seed, curandState* state) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &state[id]);
}

__global__ void GenerateGpu(curandState* state, const int64_t n, const float rate, int8_t* mask) {
  using CuInt64T = unsigned long long int;
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = state[id];
  union Pack {
    CuInt64T i_value;
    int8_t b_value[8];
  };
  CuInt64T* mask_as_int = reinterpret_cast<CuInt64T*>(mask);
  Pack pack;
  CUDA_1D_KERNEL_LOOP(i, n / 8) {
    pack.b_value[0] = curand_uniform(&localState) > rate;
    pack.b_value[1] = curand_uniform(&localState) > rate;
    pack.b_value[2] = curand_uniform(&localState) > rate;
    pack.b_value[3] = curand_uniform(&localState) > rate;
    pack.b_value[4] = curand_uniform(&localState) > rate;
    pack.b_value[5] = curand_uniform(&localState) > rate;
    pack.b_value[6] = curand_uniform(&localState) > rate;
    pack.b_value[7] = curand_uniform(&localState) > rate;
    mask_as_int[i] = pack.i_value;
  }
  const int32_t other_cnt = n % 8;
  const int32_t fast_cnt = n - other_cnt;
  if (id < other_cnt) { mask[id + fast_cnt] = curand_uniform(&localState) > rate; }
  state[id] = localState;
}

}  // namespace

RandomMaskGenerator<DeviceType::kGPU>::RandomMaskGenerator(int64_t seed, DeviceCtx* device_ctx) {
  CHECK_NOTNULL(device_ctx);
  cudaDeviceProp prop;
  CudaCheck(cudaGetDeviceProperties(&prop, 0));
  block_num_ = prop.multiProcessorCount;
  thread_num_ = 256;
  CudaCheck(cudaMalloc((void**)&curand_states_, block_num_ * thread_num_ * sizeof(curandState)));
  SetupKernel<<<block_num_, thread_num_>>>(seed, curand_states_);
}

RandomMaskGenerator<DeviceType::kGPU>::~RandomMaskGenerator() {
  CudaCheck(cudaFree(curand_states_));
}

void RandomMaskGenerator<DeviceType::kGPU>::Generate(const int64_t elem_cnt, const float rate,
                                                     int8_t* mask) {
  GenerateGpu<<<block_num_, thread_num_>>>(curand_states_, elem_cnt, rate, mask);
}

template class RandomMaskGenerator<DeviceType::kGPU>;

}  // namespace oneflow
