#include "oneflow/customized/kernels/two_stage_reduce_kernel_util.h"

namespace oneflow {

template<typename T, typename K>
struct TwoStageReduceKernelUtil<DeviceType::kCPU, T, K> {
  static void Divide(DeviceCtx* ctx, const int64_t n, const T* x, const K* count, T* y) {
    FOR_RANGE(int64_t, i, 0, n) { y[i] = x[i] / count[i]; }
  }

  static void Mask(DeviceCtx* ctx, const int64_t n, const T* x, const K* mask, T* y) {
    FOR_RANGE(int64_t, i, 0, n) { y[i] = static_cast<T>(mask[i]) * x[i]; }
  }

  static void Scale(DeviceCtx* ctx, const int64_t n, const T* x, const K* scale, T* y) {
    FOR_RANGE(int64_t, i, 0, n) { y[i] = x[i] * scale[i]; }
  }
};

#define INSTANTIATE_TWO_STAGE_REDUCE_KERNEL_UTIL_CPU(data_type_pair, index_type_pair)          \
  template struct TwoStageReduceKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                           OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_TWO_STAGE_REDUCE_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ INDEX_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_TWO_STAGE_REDUCE_KERNEL_UTIL_CPU

}  // namespace oneflow
