#ifndef ONEFLOW_CORE_KERNEL_UTIL_CPU_ARITHEMETIC_INTERFACE_H_
#define ONEFLOW_CORE_KERNEL_UTIL_CPU_ARITHEMETIC_INTERFACE_H_

#include "oneflow/core/kernel/util/arithemetic_interface.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

template<>
struct ArithemeticIf<DeviceType::kCPU> {
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                        const Shape& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const float* x, float* y);
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                        const Shape& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const double* x, double* y);
  static void Exp(DeviceCtx* ctx, const int64_t n, const float* x, float* y);
  static void Exp(DeviceCtx* ctx, const int64_t n, const double* x, double* y);
  static void InitializeWithConstConf(DeviceCtx* ctx,
                                      const ConstantInitializerConf& initializer_conf, Blob* blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UTIL_CPU_ARITHEMETIC_INTERFACE_H_
