#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BINARY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BINARY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/ndarray_apply_binary_core.h"

namespace oneflow {

template<DeviceType device_type, typename T, template<typename> class binary_func,
         typename Enable = void>
struct NdarrayApplyBinary;

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayApplyBinary<
    device_type, T, binary_func,
    typename std::enable_if<std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void Apply(DeviceCtx* ctx,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    NdarrayApplyBinaryCoreWrapper<device_type, T, binary_func>::Apply(ctx, y, a, b);
  }
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    NdarrayApplyBinaryCoreWrapper<device_type, T, binary_func>::InplaceApply(ctx, y, x);
  }
};

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayApplyBinary<
    device_type, T, binary_func,
    typename std::enable_if<!std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  using NewT = typename DevDType<device_type, T>::type;
  static void Apply(DeviceCtx* ctx,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    return NdarrayApplyBinary<device_type, NewT, binary_func>::Apply(
        ctx, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(a),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(b));
  }
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    return NdarrayApplyBinary<device_type, NewT, binary_func>::InplaceApply(
        ctx, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(x));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BINARY_H_
