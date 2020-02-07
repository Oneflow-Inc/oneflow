#include "oneflow/core/kernel/square_sum_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

template<typename T>
struct SquareSumKernelUtil<DeviceType::kCPU, T> {
  static void SquareSum(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
    T sum = 0;
    FOR_RANGE(int64_t, i, 0, n) { sum += x[i] * x[i]; }
    *y = sum;
  }
};

#define INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_CPU(type_cpp, type_proto) \
  template struct SquareSumKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_CPU, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_CPU

}  // namespace oneflow
