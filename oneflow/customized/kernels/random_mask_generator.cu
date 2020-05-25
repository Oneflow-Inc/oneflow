#include "oneflow/customized/kernels/random_mask_generator.h"

namespace oneflow {

namespace {

__global__ void SetupKernel(int64_t seed, curandState* state) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &state[id]);
}

__global__ void GenerateGpu(curandState* state, const int64_t n, const float rate, int8_t* mask) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = state[id];
  using PackType = unsigned long long int;
  union Pack {
    PackType i_value;
    int8_t b_value[sizeof(PackType)];
  };
  PackType* pack_mask = reinterpret_cast<PackType*>(mask);
  Pack pack;
  CUDA_1D_KERNEL_LOOP(i, n / sizeof(PackType)) {
#pragma unroll
    for (int j = 0; j < sizeof(PackType); j++) {
      pack.b_value[j] = curand_uniform(&localState) > rate;
    }
    pack_mask[i] = pack.i_value;
  }
  const int32_t other_cnt = n % sizeof(PackType);
  const int32_t fast_cnt = n - other_cnt;
  if (id < other_cnt) { mask[id + fast_cnt] = curand_uniform(&localState) > rate; }
  state[id] = localState;
}

}  // namespace

RandomMaskGenerator<DeviceType::kGPU>::RandomMaskGenerator(int64_t seed) {
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

void RandomMaskGenerator<DeviceType::kGPU>::Generate(DeviceCtx* device_ctx, const int64_t n,
                                                     const float rate, int8_t* mask) {
  GenerateGpu<<<block_num_, thread_num_, 0, device_ctx->cuda_stream()>>>(curand_states_, n, rate,
                                                                         mask);
}

template class RandomMaskGenerator<DeviceType::kGPU>;

}  // namespace oneflow
