#include "oneflow/customized/kernels/fused_dropout.h"

namespace oneflow {

namespace {

template<typename T>
struct DataType4;

template<>
struct DataType4<float> {
  typedef float4 type;
};

template<>
struct DataType4<double> {
  typedef double4 type;
};

// template<>
// struct DataType4<half> {
//  typedef half4 type;
//};

template<>
struct DataType4<int8_t> {
  typedef char4 type;
};

template<>
struct DataType4<int32_t> {
  typedef int4 type;
};

template<>
struct DataType4<int64_t> {
  typedef longlong4 type;
};

template<typename T4>
__device__ T4 MaskAndScale4(T4 x_val, char4 mask_val, float scale) {
  T4 y_val;
  y_val.x = x_val.x * mask_val.x * scale;
  y_val.y = x_val.y * mask_val.y * scale;
  y_val.z = x_val.z * mask_val.z * scale;
  y_val.w = x_val.w * mask_val.w * scale;
  return y_val;
}

// template<>
//__device__ half4 MaskAndScale4<half4>(half4 x_val, char4 mask_val, float scale) {
//  half4 y_val;
//  half mask_x = mask_val.x;
//  half mask_y = mask_val.y;
//  half mask_z = mask_val.z;
//  half mask_w = mask_val.w;
//  y_val.x = __hmul(__hmul(x_val.x, mask_x), scale);
//  y_val.y = __hmul(__hmul(x_val.y, mask_y), scale);
//  y_val.z = __hmul(__hmul(x_val.z, mask_z), scale);
//  y_val.w = __hmul(__hmul(x_val.w, mask_w), scale);
//  return y_val;
//}

template<typename T>
__device__ T MaskAndScale(T x_val, int8_t mask_val, float scale) {
  T y_val = x_val * mask_val * scale;
  return y_val;
}

template<>
__device__ half MaskAndScale<half>(half x_val, int8_t mask_val, float scale) {
  half one_or_zero = mask_val;
  half y_val = __hmul(__hmul(x_val, one_or_zero), scale);
  return y_val;
}

__global__ void SetupKernel(int64_t seed, int32_t thread_num, curandState* state) {
  int id = threadIdx.x + blockIdx.x * thread_num;
  curand_init(seed, id, 0, &state[id]);
}

template<typename T, typename T4>
__global__ void DropoutGpu(curandState* state, const int64_t n, const float threshold,
                           const float scale, const int32_t thread_num, const T* x, T* y,
                           int8_t* mask) {
  int id = threadIdx.x + blockIdx.x * thread_num;
  curandState localState = state[id];
  const T4* x4 = reinterpret_cast<const T4*>(x);
  T4* y4 = reinterpret_cast<T4*>(y);
  char4* mask4 = reinterpret_cast<char4*>(mask);
  char4 mask4_val;
  CUDA_1D_KERNEL_LOOP(i, n / 4) {
    mask4_val.x = curand_uniform(&localState) > threshold;
    mask4_val.y = curand_uniform(&localState) > threshold;
    mask4_val.z = curand_uniform(&localState) > threshold;
    mask4_val.w = curand_uniform(&localState) > threshold;
    T4 x4_val = x4[i];
    T4 y4_val = MaskAndScale4<T4>(x4_val, mask4_val, scale);
    y4[i] = y4_val;
    mask4[i] = mask4_val;
  }
  const int32_t other_cnt = n % 4;
  const int32_t fast_cnt = n - other_cnt;
  if (id < other_cnt) {
    int8_t mask_val = curand_uniform(&localState) > threshold;
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
  SetupKernel<<<block_num_, thread_num_>>>(seed, thread_num_, curand_states_);
}

FusedDropout<DeviceType::kGPU>::~FusedDropout() { CudaCheck(cudaFree(curand_states_)); }

template<typename T>
void FusedDropout<DeviceType::kGPU>::Dropout(const int64_t elem_cnt, const float threshold,
                                             const float scale, const T* x, T* y, int8_t* mask) {
  DropoutGpu<T, typename DataType4<T>::type><<<block_num_, thread_num_>>>(
      curand_states_, elem_cnt, threshold, scale, thread_num_, x, y, mask);
}

#define INITIATE_GPU_DROPOUT_UTIL(T, typeproto)                                           \
  template void FusedDropout<DeviceType::kGPU>::Dropout<T>(                               \
      const int64_t elem_cnt, const float threshold, const float scale, const T* x, T* y, \
      int8_t* mask);

OF_PP_FOR_EACH_TUPLE(INITIATE_GPU_DROPOUT_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
