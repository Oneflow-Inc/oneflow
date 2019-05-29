#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ndarray/ndarray_reduce_impl.h"

namespace oneflow {

template<DeviceType device_type, typename T, template<typename> class binary_func,
         typename Enable = void>
struct NdarrayReduce;

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayReduce<
    device_type, T, binary_func,
    typename std::enable_if<std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    CHECK_EQ(y.shape().NumAxes(), x.shape().NumAxes());
    if (NdarrayNoReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayNoReduce<device_type, T, binary_func>::Reduce(ctx, y, x, tmp_storage);
    } else if (NdarrayScalarReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayScalarReduce<device_type, T, binary_func>::Reduce(ctx, y, x, tmp_storage);
    } else if (NdarrayMatrixRowReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayMatrixRowReduce<device_type, T, binary_func>::Reduce(ctx, y, x, tmp_storage);
    } else if (NdarrayMatrixColReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayMatrixColReduce<device_type, T, binary_func>::Reduce(ctx, y, x, tmp_storage);
    } else {
      NdarrayDefaultReduce<device_type, T, binary_func>::Reduce(ctx, y, x, tmp_storage);
    }
  }
};

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayReduce<
    device_type, T, binary_func,
    typename std::enable_if<!std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    using NewT = typename DevDType<device_type, T>::type;
    return NdarrayReduce<device_type, NewT, binary_func>::Reduce(
        ctx, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(x),
        reinterpret_cast<const XpuVarNdarray<NewT>&>(tmp_storage));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_
