#include "oneflow/core/kernel/broadcast_binary_kernel.h"
#include "oneflow/core/ndarray/xpu_broadcast_ndarray.h"
#include "oneflow/core/ndarray/xpu_binary_func_ndarray.h"
#include "oneflow/core/ndarray/cpu_ndarray_assign.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<typename T, int NDIMS, const T (*binary_func)(const T, const T)>
struct BroadcastBinaryFunc<DeviceType::kCPU, T, NDIMS, binary_func> final {
  static void Invoke(DeviceCtx* ctx, XpuVarNdarray<T>&& y, const XpuVarNdarray<const T>& a,
                     const XpuVarNdarray<const T>& b) {
    XpuBroadcastNdarray<const T> a_broadcasted(y.shape(), a);
    XpuBroadcastNdarray<const T> b_broadcasted(y.shape(), b);
    XpuBinaryFuncNdarray<const T, binary_func> binary_func_ndarray(a_broadcasted, b_broadcasted);
    CpuNdArrayAssign<NDIMS>(&y, binary_func_ndarray);
  }
};

#define INSTANTIATE_BROADCAST_BINARY_FUNC(dtype_pair, NDIMS, binary_func)                    \
  template struct BroadcastBinaryFunc<DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS, \
                                      binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BROADCAST_BINARY_FUNC, ARITHMETIC_DATA_TYPE_SEQ,
                                 DIM_SEQ, ARITHMETIC_BINARY_FUNC_SEQ)

}  // namespace oneflow
