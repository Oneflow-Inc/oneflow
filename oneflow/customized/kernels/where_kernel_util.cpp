#include "oneflow/customized/kernels/where_kernel_util.h"

namespace oneflow {

template<typename T, typename CondT>
struct WhereKernelUtil<DeviceType::kCPU, T, CondT> {
  static void Where(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                    const T* rhs, T* out) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = static_cast<bool>(cond[i]) ? lhs[i] : rhs[i]; }
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_WHERE_FUNCTOR, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
