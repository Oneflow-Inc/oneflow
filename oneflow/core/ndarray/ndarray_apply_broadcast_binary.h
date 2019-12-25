#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_

#include "oneflow/core/ndarray/ndarray_apply_broadcast_binary_core.h"
#include "oneflow/core/ndarray/ndarray_apply_binary.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T, template<typename> class binary_func,
         typename Enable = void>
struct NdarrayApplyBroadcastBinary;

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinary<
    device_type, T, binary_func,
    typename std::enable_if<std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void Apply(DeviceCtx* ctx,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    if (a.shape() == b.shape()) {
      return NdarrayApplyBinary<device_type, T, binary_func>::Apply(ctx, y, a, b);
    }
    if (y.shape() == a.shape() && y.ptr() == a.ptr()) { return InplaceApply(ctx, y, b); }
    CheckBroadcastable(y, a, b);
    DimVector simplified_y_dim;
    DimVector simplified_a_dim;
    DimVector simplified_b_dim;
    SimplifyBroadcastShapes(y.shape(), a.shape(), b.shape(), &simplified_y_dim, &simplified_a_dim,
                            &simplified_b_dim);
    return SwitchApply(SwitchCase(simplified_y_dim.size()), ctx,
                       XpuVarNdarray<T>(Shape(simplified_y_dim), y.ptr()),
                       XpuVarNdarray<const T>(Shape(simplified_a_dim), a.ptr()),
                       XpuVarNdarray<const T>(Shape(simplified_b_dim), b.ptr()));
  }

  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    if (y.shape() == x.shape()) {
      return NdarrayApplyBinary<device_type, T, binary_func>::InplaceApply(ctx, y, x);
    }
    CheckBroadcastable(y, reinterpret_cast<const XpuVarNdarray<const T>&>(y), x);
    DimVector simplified_y_dim;
    DimVector simplified_x_dim;
    SimplifyBroadcastShapes(y.shape(), x.shape(), &simplified_y_dim, &simplified_x_dim);
    return SwitchInplaceApply(SwitchCase(simplified_y_dim.size()), ctx,
                              XpuVarNdarray<T>(Shape(simplified_y_dim), y.ptr()),
                              XpuVarNdarray<const T>(Shape(simplified_x_dim), x.ptr()));
  }

 private:
#define MAKE_NDARRAY_BROADCAST_BINARY(func_name, NDIMS) \
  NdarrayApplyBroadcastBinaryCoreWrapper<device_type, T, NDIMS, binary_func>::func_name
  DEFINE_STATIC_SWITCH_FUNC(void, Apply, MAKE_NDARRAY_BROADCAST_BINARY,
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef MAKE_NDARRAY_BROADCAST_BINARY

#define MAKE_NDARRAY_INPLACE_BROADCAST_BINARY(func_name, NDIMS) \
  NdarrayApplyBroadcastInplaceBinaryCoreWrapper<device_type, T, NDIMS, binary_func>::func_name
  DEFINE_STATIC_SWITCH_FUNC(void, InplaceApply, MAKE_NDARRAY_INPLACE_BROADCAST_BINARY,
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef MAKE_NDARRAY_INPLACE_BROADCAST_BINARY

  static void CheckBroadcastable(
      const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
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

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinary<
    device_type, T, binary_func,
    typename std::enable_if<!std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  using NewT = typename DevDType<device_type, T>::type;
  static void Apply(DeviceCtx* ctx,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    return NdarrayApplyBroadcastBinary<device_type, NewT, binary_func>::Apply(
        ctx, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(a),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(b));
  }
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    return NdarrayApplyBroadcastBinary<device_type, NewT, binary_func>::InplaceApply(
        ctx, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(x));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_
