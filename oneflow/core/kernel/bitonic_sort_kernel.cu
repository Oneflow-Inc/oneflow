#include "oneflow/core/kernel/bitonic_sort_kernel.h"
#include "oneflow/core/kernel/gpu_bitonic_sort.cuh"

namespace oneflow {

namespace {

template<typename T>
struct LTComp {
  __device__ bool operator()(const T& x, const T& y) { return x < y; }
};

template<typename T>
struct GTComp {
  __device__ bool operator()(const T& x, const T& y) { return x > y; }
};

template<typename T>
__global__ void GpuForward(const int32_t instance_size, T* out) {
  bitonicSort<T, LTComp<T>>(out + blockIdx.x * instance_size, instance_size);
}

}  // namespace

template<typename T>
struct BitonicSortUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const int32_t instance_num, const int32_t instance_size,
                      T* out) {
    GpuForward<<<instance_num, 1, 0, ctx->cuda_stream()>>>(instance_size, out);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) \
  template struct BitonicSortUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
