#include "oneflow/core/ndarray/ndarray_apply_unary_core.h"
#include "oneflow/core/ndarray/unary_func.h"

namespace oneflow {

template<typename T, const T (*unary_func)(const T)>
struct NdArrayApplyUnaryCoreWrapper<DeviceType::kCPU, T, unary_func> final {
  static void ImplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y) {
    NdArrayApplyUnaryCore<T, unary_func>::ImplaceApply(y.ptr(), y.shape().ElemNum());
  }
};

#define INSTANTIATE_NDARRAY_APPLY_UNARY_CORE(dtype_pair, unary_func)                           \
  template struct NdArrayApplyUnaryCoreWrapper<DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype_pair), \
                                               unary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_APPLY_UNARY_CORE, ARITHMETIC_DATA_TYPE_SEQ,
                                 ARITHMETIC_UNARY_FUNC_SEQ)

}  // namespace oneflow
