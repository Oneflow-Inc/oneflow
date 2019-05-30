#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/ndarray_apply_unary_core.h"

namespace oneflow {

template<DeviceType device_type, typename T, template<typename> class unary_func,
         typename Enable = void>
struct NdarrayApplyUnary;

template<DeviceType device_type, typename T, template<typename> class unary_func>
struct NdarrayApplyUnary<
    device_type, T, unary_func,
    typename std::enable_if<std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y) {
    NdarrayApplyUnaryCoreWrapper<device_type, T, unary_func>::InplaceApply(ctx, y);
  }
};

template<DeviceType device_type, typename T, template<typename> class unary_func>
struct NdarrayApplyUnary<
    device_type, T, unary_func,
    typename std::enable_if<!std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y) {
    using NewT = typename DevDType<device_type, T>::type;
    return NdarrayApplyUnary<device_type, NewT, unary_func>::InplaceApply(
        ctx, reinterpret_cast<const XpuVarNdarray<NewT>&>(y));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_H_
