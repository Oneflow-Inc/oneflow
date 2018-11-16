#include "oneflow/core/ndarray/ndarray_reduce_core.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T, int NDIMS>
struct NdArrayReduceCoreWrapper<DeviceType::kCPU, T, NDIMS> final {
  static void ReduceAxis(T* dst_ptr, const XpuVarNdarray<const T>& x, int axis,
                         int64_t new_dim_value) {
    NdArrayReduceCore<T, NDIMS>::ReduceAxis(dst_ptr, x, axis, new_dim_value);
  }
  static void ImplaceReduceAxis(const XpuReducedNdarray<T, NDIMS>& x, int axis,
                                int64_t new_dim_value) {
    NdArrayReduceCore<T, NDIMS>::ImplaceReduceAxis(x, axis, new_dim_value);
  }
};

#define INSTANTIATE_NDARRAY_REDUCE(dtype_pair, NDIMS) \
  template struct NdArrayReduceCoreWrapper<DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE, ARITHMETIC_DATA_TYPE_SEQ, DIM_SEQ);

}  // namespace oneflow
