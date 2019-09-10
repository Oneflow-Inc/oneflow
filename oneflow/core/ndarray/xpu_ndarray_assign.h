#ifndef ONEFLOW_CORE_NDARRAY_XPU_ASSIGN_H_
#define ONEFLOW_CORE_NDARRAY_XPU_ASSIGN_H_

#include "oneflow/core/ndarray/ndarray_assign_core.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename Enable = void>
struct XpuNdarrayAssign;

template<DeviceType device_type, typename T>
struct XpuNdarrayAssign<
    device_type, T,
    typename std::enable_if<std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  template<int NDIMS>
  static void Assign(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                     const XpuReducedNdarray<T, NDIMS>& reduced) {
    NdarrayAssignCoreWrapper<device_type, T, NDIMS>::Assign(ctx, y, reduced);
  }
  static void Assign(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    CHECK(y.shape() == x.shape());
    if (x.ptr() == y.ptr()) { return; }
    Memcpy<device_type>(ctx, y.ptr(), x.ptr(), y.shape().ElemNum() * sizeof(T));
  }
};

template<DeviceType device_type, typename T>
struct XpuNdarrayAssign<
    device_type, T,
    typename std::enable_if<!std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  using NewT = typename DevDType<device_type, T>::type;
  template<int NDIMS>
  static void Assign(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                     const XpuReducedNdarray<T, NDIMS>& reduced) {
    XpuNdarrayAssign<device_type, NewT>::Assign(
        ctx, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuReducedNdarray<NewT, NDIMS>&>(reduced));
  }

  static void Assign(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    XpuNdarrayAssign<device_type, NewT>::Assign(
        ctx, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(x));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_ASSIGN_H_
