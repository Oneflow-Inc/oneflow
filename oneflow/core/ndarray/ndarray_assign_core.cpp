#include "oneflow/core/ndarray/ndarray_assign_core.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T, int NDIMS>
struct NdArrayAssignCoreWrapper<DeviceType::kCPU, T, NDIMS> final {
  static void Assign(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                     const XpuReducedNdarray<T, NDIMS>& reduced) {
    NdArrayAssignCore<T, NDIMS>::Assign(y, reduced);
  }
};

#define INSTANTIATE_NDARRAY_ASSIGN(dtype_pair, NDIMS) \
  template struct NdArrayAssignCoreWrapper<DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_ASSIGN, ARITHMETIC_DATA_TYPE_SEQ, DIM_SEQ);

}  // namespace oneflow
