#include "oneflow/core/ndarray/ndarray_reduce_core.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T, int NDIMS, const T (*binary_func)(const T, const T)>
struct NdArrayReduceCoreWrapper<DeviceType::kCPU, T, NDIMS, binary_func> final {
  static void ReduceAxis(DeviceCtx* ctx, const XpuReducedNdarray<T, NDIMS>& dst_reduced,
                         const XpuReducedNdarray<T, NDIMS>& x, int axis) {
    NdArrayReduceCore<T, NDIMS, binary_func>::ReduceAxis(dst_reduced, x, axis);
  }
};

#define INSTANTIATE_NDARRAY_REDUCE(dtype_pair, NDIMS, binary_func)                                \
  template struct NdArrayReduceCoreWrapper<DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS, \
                                           binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE, ARITHMETIC_DATA_TYPE_SEQ, DIM_SEQ,
                                 REDUCE_BINARY_FUNC_SEQ);

}  // namespace oneflow
