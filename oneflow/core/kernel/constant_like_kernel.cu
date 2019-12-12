#include "oneflow/core/kernel/constant_like_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GpuForward(const int64_t elem_cnt, const T scalar, T* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { out_ptr[i] = scalar; }
}

}  // namespace

template<typename T>
struct ConstantLikeUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const T scalar, T* out_ptr) {
    GpuForward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        elem_cnt, scalar, out_ptr);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) \
  template struct ConstantLikeUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow