#include "oneflow/customized/kernels/fused_dropout.h"

namespace oneflow {

namespace {

struct half4 {
  half x;
  half y;
  half z;
  half w;
};

template<typename T>
struct Vec4Type;

template<>
struct Vec4Type<float> {
  typedef float4 type;
};

template<>
struct Vec4Type<double> {
  typedef double4 type;
};

template<>
struct Vec4Type<half> {
  typedef half4 type;
};

template<>
struct Vec4Type<int8_t> {
  typedef char4 type;
};

template<>
struct Vec4Type<int32_t> {
  typedef int4 type;
};

template<>
struct Vec4Type<int64_t> {
  typedef longlong4 type;
};

template<typename T>
__device__ T MaskAndScale(T x, int8_t mask, float scale) {
  return x * mask * scale;
}

template<>
__device__ half MaskAndScale<half>(half x, int8_t mask, float scale) {
  return __hmul(x, __float2half(mask * scale));
}

__global__ void SetupKernel(int64_t seed, curandState* state) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &state[id]);
}

template<typename T>
__global__ void DropoutGpu(curandState* state, const int64_t n, const float rate, const float scale,
                           const T* x, T* y, int8_t* mask) {
  using Vec4 = typename Vec4Type<T>::type;
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = state[id];
  const Vec4* x4 = reinterpret_cast<const Vec4*>(x);
  Vec4* y4 = reinterpret_cast<Vec4*>(y);
  char4* mask4 = reinterpret_cast<char4*>(mask);
  char4 mask4_val;
  CUDA_1D_KERNEL_LOOP(i, n / 4) {
    mask4_val.x = curand_uniform(&localState) > rate;
    mask4_val.y = curand_uniform(&localState) > rate;
    mask4_val.z = curand_uniform(&localState) > rate;
    mask4_val.w = curand_uniform(&localState) > rate;
    Vec4 x4_val = x4[i];
    Vec4 y4_val;
    y4_val.x = MaskAndScale<T>(x4_val.x, mask4_val.x, scale);
    y4_val.y = MaskAndScale<T>(x4_val.y, mask4_val.y, scale);
    y4_val.z = MaskAndScale<T>(x4_val.z, mask4_val.z, scale);
    y4_val.w = MaskAndScale<T>(x4_val.w, mask4_val.w, scale);
    y4[i] = y4_val;
    mask4[i] = mask4_val;
  }
  const int32_t other_cnt = n % 4;
  const int32_t fast_cnt = n - other_cnt;
  if (id < other_cnt) {
    int8_t mask_val = curand_uniform(&localState) > rate;
    y[id + fast_cnt] = MaskAndScale<T>(x[id + fast_cnt], mask_val, scale);
    mask[id + fast_cnt] = mask_val;
  }
  state[id] = localState;
}

}  // namespace

FusedDropout<DeviceType::kGPU>::FusedDropout(int64_t seed, DeviceCtx* device_ctx) {
  CHECK_NOTNULL(device_ctx);
  cudaDeviceProp prop;
  CudaCheck(cudaGetDeviceProperties(&prop, 0));
  block_num_ = prop.multiProcessorCount;
  thread_num_ = 256;
  CudaCheck(cudaMalloc((void**)&curand_states_, block_num_ * thread_num_ * sizeof(curandState)));
  SetupKernel<<<block_num_, thread_num_>>>(seed, curand_states_);
}

FusedDropout<DeviceType::kGPU>::~FusedDropout() { CudaCheck(cudaFree(curand_states_)); }

template<typename T>
void FusedDropout<DeviceType::kGPU>::Dropout(const int64_t elem_cnt, const float rate,
                                             const float scale, const T* x, T* y, int8_t* mask) {
  DropoutGpu<T><<<block_num_, thread_num_>>>(curand_states_, elem_cnt, rate, scale, x, y, mask);
}

template<>
void FusedDropout<DeviceType::kGPU>::Dropout<float16>(const int64_t elem_cnt, const float rate,
                                                      const float scale, const float16* x,
                                                      float16* y, int8_t* mask) {
  DropoutGpu<half><<<block_num_, thread_num_>>>(curand_states_, elem_cnt, rate, scale,
                                                reinterpret_cast<const half*>(x),
                                                reinterpret_cast<half*>(y), mask);
}

#define INITIATE_GPU_FUSED_DROPOUT(T, typeproto)                                                \
  template void FusedDropout<DeviceType::kGPU>::Dropout<T>(const int64_t elem_cnt,              \
                                                           const float rate, const float scale, \
                                                           const T* x, T* y, int8_t* mask);

OF_PP_FOR_EACH_TUPLE(INITIATE_GPU_FUSED_DROPOUT, ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);

}  // namespace oneflow
