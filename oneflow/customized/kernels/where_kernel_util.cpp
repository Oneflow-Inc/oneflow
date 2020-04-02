#include "oneflow/customized/kernels/where_kernel_util.h"

namespace oneflow {

template<typename T, typename CondT>
struct WhereFunctor<DeviceType::kCPU, T, CondT> {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                  const T* rhs, T* out) const {
    DoWhere(elem_cnt, cond, lhs, rhs, out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_WHERE_FUNCTOR, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
