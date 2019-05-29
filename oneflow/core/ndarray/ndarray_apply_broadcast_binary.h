#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_

#include "oneflow/core/ndarray/ndarray_apply_broadcast_binary_core.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS, template<typename> class binary_func,
         typename Enable = void>
struct NdarrayApplyBroadcastBinary;

template<DeviceType device_type, typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinary<
    device_type, T, NDIMS, binary_func,
    typename std::enable_if<std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void Apply(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& a,
                    const XpuVarNdarray<const T>& b) {
    using NdarrayAssign = XpuNdarrayAssign<device_type, T>;
    using BroadcastBinary =
        NdarrayApplyBroadcastBinaryCoreWrapper<device_type, T, NDIMS, binary_func>;
    CheckBroadcastable(y, a, b);
    return BroadcastBinary::Apply(ctx, y, a, b);
    if (a.shape() == y.shape()) {
      NdarrayAssign::Assign(ctx, y, a);
      BroadcastBinary::ImplaceApply(ctx, y, b);
    } else if (b.shape() == y.shape()) {
      NdarrayAssign::Assign(ctx, y, b);
      BroadcastBinary::ImplaceApply(ctx, y, a);
    } else {
      BroadcastBinary::Apply(ctx, y, a, b);
    }
  }

 private:
  static void CheckBroadcastable(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& a,
                                 const XpuVarNdarray<const T>& b) {
    CHECK_EQ(y.shape().NumAxes(), a.shape().NumAxes());
    CHECK_EQ(y.shape().NumAxes(), b.shape().NumAxes());
    for (int i = 0; i < y.shape().NumAxes(); ++i) {
      CHECK_EQ(y.shape().At(i), std::max(a.shape().At(i), b.shape().At(i)));
      if (a.shape().At(i) != b.shape().At(i)) {
        CHECK(a.shape().At(i) == 1 || b.shape().At(i) == 1);
      }
    }
  }
};

template<DeviceType device_type, typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinary<
    device_type, T, NDIMS, binary_func,
    typename std::enable_if<!std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void Apply(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& a,
                    const XpuVarNdarray<const T>& b) {
    using NewT = typename DevDType<device_type, T>::type;
    return NdarrayApplyBroadcastBinary<device_type, NewT, NDIMS, binary_func>::Apply(
        ctx, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(a),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(b));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_
