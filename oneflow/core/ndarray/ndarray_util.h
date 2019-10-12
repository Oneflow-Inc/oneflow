#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_UTIL_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_var_ndarray_builder.h"
#include "oneflow/core/ndarray/ndarray_reduce.h"
#include "oneflow/core/ndarray/ndarray_apply_unary.h"
#include "oneflow/core/ndarray/ndarray_apply_binary.h"
#include "oneflow/core/ndarray/ndarray_apply_broadcast_unary.h"
#include "oneflow/core/ndarray/ndarray_apply_broadcast_binary.h"
#include "oneflow/core/ndarray/xpu_reduced_ndarray.h"
#include "oneflow/core/ndarray/xpu_ndarray_assign.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct NdarrayUtil final {
  static XpuVarNdarrayBuilder<const T> GetValNdarrayBuilder() {
    return XpuVarNdarrayBuilder<const T>();
  }
  static XpuVarNdarrayBuilder<T> GetVarNdarrayBuilder() { return XpuVarNdarrayBuilder<T>(); }

  static void Assign(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    return XpuNdarrayAssign<device_type, T>::Assign(ctx, y, x);
  }

  static void BroadcastTo(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                          const XpuVarNdarray<const T>& x) {
    return BroadcastIdentity(ctx, y, x);
  }

#define DEFINE_UNARY_FUNC(func_name)                                                         \
  static void func_name(                                                                     \
      DeviceCtx* ctx,                                                                        \
      const XpuVarNdarray<typename UnaryFuncTrait<UnaryFunc##func_name, T>::return_type>& y, \
      const XpuVarNdarray<const T>& x) {                                                     \
    return ApplyUnary<UnaryFunc##func_name>(ctx, y, x);                                      \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_UNARY_FUNC, ARITHMETIC_UNARY_FUNC_NAME_SEQ)
#undef DEFINE_UNARY_FUNC

#define DEFINE_BINARY_FUNC(func_name)                                                          \
  static void func_name(                                                                       \
      DeviceCtx* ctx,                                                                          \
      const XpuVarNdarray<typename BinaryFuncTrait<BinaryFunc##func_name, T>::return_type>& y, \
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {                      \
    return ApplyBinary<BinaryFunc##func_name>(ctx, y, a, b);                                   \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_BINARY_FUNC, BINARY_FUNC_NAME_SEQ)
#undef DEFINE_BINARY_FUNC

#define DEFINE_BROADCAST_UNARY_FUNC(func_name)                                               \
  static void Broadcast##func_name(                                                          \
      DeviceCtx* ctx,                                                                        \
      const XpuVarNdarray<typename UnaryFuncTrait<UnaryFunc##func_name, T>::return_type>& y, \
      const XpuVarNdarray<const T>& x) {                                                     \
    return BroadcastApplyUnary<UnaryFunc##func_name>(ctx, y, x);                             \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_BROADCAST_UNARY_FUNC, ARITHMETIC_UNARY_FUNC_NAME_SEQ)
#undef DEFINE_BROADCAST_UNARY_FUNC

#define DEFINE_BROADCAST_BINARY_FUNC(func_name)                                                \
  static void Broadcast##func_name(                                                            \
      DeviceCtx* ctx,                                                                          \
      const XpuVarNdarray<typename BinaryFuncTrait<BinaryFunc##func_name, T>::return_type>& y, \
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {                      \
    return BroadcastApplyBinary<BinaryFunc##func_name>(ctx, y, a, b);                          \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_BROADCAST_BINARY_FUNC, BINARY_FUNC_NAME_SEQ)
#undef DEFINE_BROADCAST_BINARY_FUNC

#define DEFINE_INPLACE_UNARY_FUNC(func_name)                                  \
  static void Inplace##func_name(DeviceCtx* ctx, const XpuVarNdarray<T>& y) { \
    InplaceApply<UnaryFunc##func_name>(ctx, y);                               \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_INPLACE_UNARY_FUNC, ARITHMETIC_UNARY_FUNC_NAME_SEQ)
#undef DEFINE_INPLACE_UNARY_FUNC

#define DEFINE_INPLACE_BINARY_FUNC(func_name)                               \
  static void Inplace##func_name(DeviceCtx* ctx, const XpuVarNdarray<T>& y, \
                                 const XpuVarNdarray<const T>& x) {         \
    InplaceApply<BinaryFunc##func_name>(ctx, y, x);                         \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_INPLACE_BINARY_FUNC, ARITHMETIC_BINARY_FUNC_NAME_SEQ)
#undef DEFINE_INPLACE_BINARY_FUNC

#define DEFINE_INPLACE_BROADCAST_BINARY_FUNC(func_name)                              \
  static void InplaceBroadcast##func_name(DeviceCtx* ctx, const XpuVarNdarray<T>& y, \
                                          const XpuVarNdarray<const T>& x) {         \
    return InplaceBroadcastApply<BinaryFunc##func_name>(ctx, y, x);                  \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_INPLACE_BROADCAST_BINARY_FUNC, ARITHMETIC_BINARY_FUNC_NAME_SEQ)
#undef DEFINE_INPLACE_BROADCAST_BINARY_FUNC

#define DEFINE_REDUCE_FUNC(func_name)                                                            \
  static void Reduce##func_name(DeviceCtx* ctx, const XpuVarNdarray<T>& y,                       \
                                const XpuVarNdarray<const T>& x,                                 \
                                const XpuVarNdarray<T>& tmp_storage) {                           \
    return NdarrayReduce<device_type, T, BinaryFunc##func_name>::Reduce(ctx, y, x, tmp_storage); \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_REDUCE_FUNC, REDUCE_BINARY_FUNC_NAME_SEQ)
#undef DEFINE_REDUCE_FUNC

 private:
  template<template<typename> class unary_func>
  static void BroadcastApplyUnary(
      DeviceCtx* ctx, const XpuVarNdarray<typename UnaryFuncTrait<unary_func, T>::return_type>& y,
      const XpuVarNdarray<const T>& x) {
    CHECK_EQ(x.shape().NumAxes(), y.shape().NumAxes());
    return Unary<unary_func>::SwitchBroadcastApply(SwitchCase(x.shape().NumAxes()), ctx, y, x);
  }

  template<template<typename> class binary_func>
  static void BroadcastApplyBinary(
      DeviceCtx* ctx, const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    CHECK_EQ(a.shape().NumAxes(), y.shape().NumAxes());
    CHECK_EQ(b.shape().NumAxes(), y.shape().NumAxes());
    int64_t num_axes = y.shape().NumAxes();
    std::vector<int64_t> ndims_list(num_axes);
    ndims_list.at(0) = 0;

    auto IsXpuShapeContinuesOne = [&](const XpuShape& shape, int64_t index) -> bool {
      CHECK(index > 0 && index < shape.NumAxes());
      return shape.At(index) == 1 && shape.At(index - 1) == 1;
    };

    auto IsXpuShapeBothContinuesMultiAndEqual = [&](const XpuShape& lhs, const XpuShape& rhs,
                                                    int64_t index) -> bool {
      CHECK(index > 0 && index < lhs.NumAxes() && index < rhs.NumAxes());
      return lhs.At(index) > 1 && lhs.At(index - 1) > 1 && lhs.At(index) == rhs.At(index)
             && lhs.At(index - 1) == rhs.At(index - 1);
    };

    for (int64_t i = 1; i < num_axes; ++i) {
      if (IsXpuShapeContinuesOne(a.shape(), i) || IsXpuShapeContinuesOne(b.shape(), i)
          || IsXpuShapeBothContinuesMultiAndEqual(a.shape(), b.shape(), i)) {
        ndims_list.at(i) = ndims_list.at(i - 1);
      } else {
        ndims_list.at(i) = ndims_list.at(i - 1) + 1;
      }
    }

    int64_t merged_num_axes = ndims_list.back() + 1;
    CHECK_LE(merged_num_axes, num_axes);

    if (merged_num_axes == num_axes) {
      return Binary<binary_func>::SwitchBroadcastApply(SwitchCase(y.shape().NumAxes()), ctx, y, a,
                                                       b);

    } else {
      std::vector<int64_t> shape_a(merged_num_axes);
      std::vector<int64_t> shape_b(merged_num_axes);
      std::vector<int64_t> shape_y(merged_num_axes);
      int64_t merged_i = 0;
      for (int64_t i = 0; i < num_axes; ++i) {
        if (i == 0) {
          shape_a.at(0) = a.shape().At(0);
          shape_b.at(0) = b.shape().At(0);
          shape_y.at(0) = y.shape().At(0);
          continue;
        }
        if (ndims_list.at(i) == ndims_list.at(i - 1)) {
          shape_a.at(merged_i) *= a.shape().At(i);
          shape_b.at(merged_i) *= b.shape().At(i);
          shape_y.at(merged_i) *= y.shape().At(i);
        } else {
          ++merged_i;
          shape_a.at(merged_i) = a.shape().At(i);
          shape_b.at(merged_i) = b.shape().At(i);
          shape_y.at(merged_i) = y.shape().At(i);
        }
      }
      CHECK_EQ(merged_i, merged_num_axes - 1);
      XpuVarNdarray<const T> reshape_a(Shape(shape_a), a.host_ptr());
      XpuVarNdarray<const T> reshape_b(Shape(shape_b), b.host_ptr());
      XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type> reshape_y(Shape(shape_y),
                                                                                     y.host_ptr());

      return Binary<binary_func>::SwitchBroadcastApply(SwitchCase(reshape_y.shape().NumAxes()), ctx,
                                                       reshape_y, reshape_a, reshape_b);
    }
  }

  template<template<typename> class binary_func>
  static void InplaceBroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                                    const XpuVarNdarray<const T>& x) {
    static_assert(std::is_same<T, typename BinaryFuncTrait<binary_func, T>::return_type>::value,
                  "T must be same with BinaryFuncTrait<binary_func, T>::return_type");
    CHECK_EQ(x.shape().NumAxes(), y.shape().NumAxes());
    return Binary<binary_func>::SwitchInplaceBroadcastApply(SwitchCase(y.shape().NumAxes()), ctx, y,
                                                            x);
  }

  template<template<typename> class unary_func>
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y) {
    static_assert(std::is_same<T, typename UnaryFuncTrait<unary_func, T>::return_type>::value,
                  "T must be same with UnaryFuncTrait<unary_func, T>::return_type");
    return NdarrayApplyUnary<device_type, T, unary_func>::InplaceApply(ctx, y);
  }

  template<template<typename> class binary_func>
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    static_assert(std::is_same<T, typename BinaryFuncTrait<binary_func, T>::return_type>::value,
                  "T must be same with BinaryFuncTrait<binary_func, T>::return_type");
    return NdarrayApplyBinary<device_type, T, binary_func>::InplaceApply(ctx, y, x);
  }

  template<template<typename> class unary_func>
  static void ApplyUnary(
      DeviceCtx* ctx, const XpuVarNdarray<typename UnaryFuncTrait<unary_func, T>::return_type>& y,
      const XpuVarNdarray<const T>& x) {
    return NdarrayApplyUnary<device_type, T, unary_func>::Apply(ctx, y, x);
  }

  template<template<typename> class binary_func>
  static void ApplyBinary(
      DeviceCtx* ctx, const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    return NdarrayApplyBinary<device_type, T, binary_func>::Apply(ctx, y, a, b);
  }

  template<template<typename> class unary_func>
  struct Unary final {
    template<int NDIMS>
    static void BroadcastApply(
        DeviceCtx* ctx, const XpuVarNdarray<typename UnaryFuncTrait<unary_func, T>::return_type>& y,
        const XpuVarNdarray<const T>& x) {
      return NdarrayApplyBroadcastUnary<device_type, T, NDIMS, unary_func>::Apply(ctx, y, x);
    }
#define DEFINE_NDARRAY_BROADCAST_UNARY(func_name, NDIMS) Unary<unary_func>::func_name<NDIMS>
    DEFINE_STATIC_SWITCH_FUNC(void, BroadcastApply, DEFINE_NDARRAY_BROADCAST_UNARY,
                              MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef DEFINE_NDARRAY_BROADCAST_UNARY
  };

  template<template<typename> class binary_func>
  struct Binary final {
    template<int NDIMS>
    static void BroadcastApply(
        DeviceCtx* ctx,
        const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
        const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
      return NdarrayApplyBroadcastBinary<device_type, T, NDIMS, binary_func>::Apply(ctx, y, a, b);
    }
    template<int NDIMS>
    static void InplaceBroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                                      const XpuVarNdarray<const T>& x) {
      return NdarrayApplyBroadcastBinary<device_type, T, NDIMS, binary_func>::InplaceApply(ctx, y,
                                                                                           x);
    }
#define MAKE_NDARRAY_BROADCAST_BINARY(func_name, NDIMS) Binary<binary_func>::func_name<NDIMS>
    DEFINE_STATIC_SWITCH_FUNC(void, BroadcastApply, MAKE_NDARRAY_BROADCAST_BINARY,
                              MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef MAKE_NDARRAY_BROADCAST_BINARY

#define MAKE_NDARRAY_INPLACE_BROADCAST_BINARY(func_name, NDIMS) \
  Binary<binary_func>::func_name<NDIMS>
    DEFINE_STATIC_SWITCH_FUNC(void, InplaceBroadcastApply, MAKE_NDARRAY_INPLACE_BROADCAST_BINARY,
                              MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef MAKE_NDARRAY_INPLACE_BROADCAST_BINARY
  };
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_UTIL_H_
