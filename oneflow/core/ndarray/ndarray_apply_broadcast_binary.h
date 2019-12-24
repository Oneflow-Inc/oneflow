#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_

#include "oneflow/core/ndarray/ndarray_apply_broadcast_binary_core.h"
#include "oneflow/core/ndarray/ndarray_apply_binary.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void SimplifyBroadcastBinaryShapes(const XpuShape& y, const XpuShape& b, DimVector* simplified_y,
                                   DimVector* simplified_b);
void SimplifyBroadcastBinaryShapes(const XpuShape& y, const XpuShape& a, const XpuShape& b,
                                   DimVector* simplified_y, DimVector* simplified_a,
                                   DimVector* simplified_b);
template<DeviceType device_type, typename T, int NDIMS, template<typename> class binary_func,
         typename Enable = void>
struct NdarrayApplyBroadcastBinary;

template<DeviceType device_type, typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinary<
    device_type, T, NDIMS, binary_func,
    typename std::enable_if<std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void Apply(DeviceCtx* ctx,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    if (a.shape() == b.shape()) {
      return NdarrayApplyBinary<device_type, T, binary_func>::Apply(ctx, y, a, b);
    }
    using BroadcastBinary =
        NdarrayApplyBroadcastBinaryCoreWrapper<device_type, T, NDIMS, binary_func>;
    CheckBroadcastable(y, a, b);
    DimVector simplified_y_dim;
    DimVector simplified_a_dim;
    DimVector simplified_b_dim;
    SimplifyBroadcastBinaryShapes(y.shape(), a.shape(), b.shape(), &simplified_y_dim,
                                  &simplified_a_dim, &simplified_b_dim);
    return BroadcastBinary::Apply(ctx, XpuVarNdarray<T>(Shape(simplified_y_dim), y.ptr()),
                                  XpuVarNdarray<const T>(Shape(simplified_a_dim), a.ptr()),
                                  XpuVarNdarray<const T>(Shape(simplified_b_dim), b.ptr()));
  }

  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    if (y.shape() == x.shape()) {
      return NdarrayApplyBinary<device_type, T, binary_func>::InplaceApply(ctx, y, x);
    }
    using BroadcastBinary =
        NdarrayApplyBroadcastBinaryCoreWrapper<device_type, T, NDIMS, binary_func>;
    CheckBroadcastable(y, reinterpret_cast<const XpuVarNdarray<const T>&>(y), x);
    DimVector simplified_y_dim;
    DimVector simplified_x_dim;
    SimplifyBroadcastBinaryShapes(y.shape(), x.shape(), &simplified_y_dim, &simplified_x_dim);
    return BroadcastBinary::InplaceApply(ctx, XpuVarNdarray<T>(Shape(simplified_y_dim), y.ptr()),
                                         XpuVarNdarray<const T>(Shape(simplified_x_dim), x.ptr()));
  }

 private:
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

template<DeviceType device_type, typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinary<
    device_type, T, NDIMS, binary_func,
    typename std::enable_if<!std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  using NewT = typename DevDType<device_type, T>::type;
  static void Apply(DeviceCtx* ctx,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    return NdarrayApplyBroadcastBinary<device_type, NewT, NDIMS, binary_func>::Apply(
        ctx, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(a),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(b));
  }
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    return NdarrayApplyBroadcastBinary<device_type, NewT, NDIMS, binary_func>::InplaceApply(
        ctx, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(x));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_
