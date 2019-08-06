#include "oneflow/core/ndarray/ndarray_apply_binary_core.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<typename T, template<typename> class binary_func>
struct NdarrayApplyBinaryCoreWrapper<DeviceType::kCPU, T, binary_func> final {
  static void Apply(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& a,
                    const XpuVarNdarray<const T>& b) {
    NdarrayApplyBinaryCore<T, binary_func>::Apply(y.shape().ElemNum(), y.ptr(), a.ptr(), b.ptr());
  }
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    NdarrayApplyBinaryCore<T, binary_func>::InplaceApply(y.shape().ElemNum(), y.ptr(), x.ptr());
  }
};

#define INSTANTIATE_NDARRAY_APPLY_BINARY_CORE(dtype_pair, binary_func)                          \
  template struct NdarrayApplyBinaryCoreWrapper<DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype_pair), \
                                                binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_APPLY_BINARY_CORE,
                                 ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ,
                                 ARITHMETIC_BINARY_FUNC_SEQ)

}  // namespace oneflow
