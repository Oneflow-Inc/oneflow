#include "oneflow/core/kernel/logical_and_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GpuForward(const int64_t elem_cnt, const T* lhs_ptr, const T* rhs_ptr, T* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    out_ptr[i] = static_cast<T>(static_cast<bool>(lhs_ptr[i]) & static_cast<bool>(rhs_ptr[i]));
  }
}

}  // namespace

template<typename T>
struct LogicalAndUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const T* lhs_ptr, const T* rhs_ptr,
                      T* out_ptr) {
    GpuForward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        elem_cnt, lhs_ptr, rhs_ptr, out_ptr);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) template struct LogicalAndUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
