#include "oneflow/customized/kernels/dropout_util.h"

namespace oneflow {

namespace {

__global__ void setup_kernel(int64_t seed, int32_t thread_num, curandState* state) {
  int id = threadIdx.x + blockIdx.x * thread_num;
  curand_init(seed, id, 0, &state[id]);
}

template<typename T>
__global__ void MaskAndScaleGpu(curandState* state, const int64_t n, const float threshold,
                                const float scale, const int32_t thread_num, const T* x, T* y,
                                int8_t* mask) {}

template<>
__global__ void MaskAndScaleGpu<float>(curandState* state, const int64_t n, const float threshold,
                                       const float scale, const int32_t thread_num, const float* x,
                                       float* y, int8_t* mask) {
  int id = threadIdx.x + blockIdx.x * thread_num;
  curandState localState = state[id];
  union Pack {
    int32_t i_value;
    int8_t b_value[4];
  };
  int32_t* mask_as_int = reinterpret_cast<int32_t*>(mask);
  Pack pack;
  float4* x4 = (float4*)x;
  float4* y4 = (float4*)y;
  CUDA_1D_KERNEL_LOOP(i, n / 4) {
    pack.b_value[0] = curand_uniform(&localState) > threshold;
    pack.b_value[1] = curand_uniform(&localState) > threshold;
    pack.b_value[2] = curand_uniform(&localState) > threshold;
    pack.b_value[3] = curand_uniform(&localState) > threshold;
    float4 xx4 = x4[i];
    float4 yy4;
    yy4.x = xx4.x * pack.b_value[0] * scale;
    yy4.y = xx4.y * pack.b_value[1] * scale;
    yy4.z = xx4.z * pack.b_value[2] * scale;
    yy4.w = xx4.w * pack.b_value[3] * scale;
    mask_as_int[i] = pack.i_value;
    y4[i] = yy4;
  }
  const int32_t other_cnt = n % 4;
  const int32_t fast_cnt = n - other_cnt;
  if (id < other_cnt) { y[id + fast_cnt] = x[id + fast_cnt]; }
  state[id] = localState;
}

}  // namespace

DropoutUtil<DeviceType::kGPU>::DropoutUtil(int64_t seed, DeviceCtx* device_ctx) {
  CHECK_NOTNULL(device_ctx);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int32_t sm_count = prop.multiProcessorCount;
  const int32_t thread_num = 256;
  cudaMalloc((void**)&curand_states_, sm_count * thread_num * sizeof(curandState));
  setup_kernel<<<sm_count, thread_num>>>(seed, thread_num, curand_states_);
}

DropoutUtil<DeviceType::kGPU>::~DropoutUtil() { CudaCheck(cudaFree(curand_states_)); }

template<typename T>
void DropoutUtil<DeviceType::kGPU>::Dropout(const int64_t elem_cnt, const float threshold,
                                            const float scale, const T* x, T* y, int8_t* mask) {
  const int32_t sm_count = 68;
  const int32_t thread_num = 256;
  MaskAndScaleGpu<T><<<sm_count, thread_num>>>(curand_states_, elem_cnt, threshold, scale,
                                               thread_num, x, y, mask);
}

#define INITIATE_GPU_DROPOUT_UTIL(T, typeproto)                                           \
  template void DropoutUtil<DeviceType::kGPU>::Dropout<T>(                                \
      const int64_t elem_cnt, const float threshold, const float scale, const T* x, T* y, \
      int8_t* mask);

OF_PP_FOR_EACH_TUPLE(INITIATE_GPU_DROPOUT_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
