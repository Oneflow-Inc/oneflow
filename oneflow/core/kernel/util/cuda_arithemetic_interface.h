#ifndef ONEFLOW_CORE_KERNEL_UTIL_CUDA_ARITHEMETIC_INTERFACE_H_
#define ONEFLOW_CORE_KERNEL_UTIL_CUDA_ARITHEMETIC_INTERFACE_H_

#include "oneflow/core/kernel/util/arithemetic_interface.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

class Blob;
class ConstantInitializerConf;

template<>
struct ArithemeticIf<DeviceType::kGPU> {
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const DenseShapeView& x_shape,
                        const DenseShapeView& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const float* x, float* y);
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const DenseShapeView& x_shape,
                        const DenseShapeView& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const double* x, double* y);
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const DenseShapeView& x_shape,
                        const DenseShapeView& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const float16* x, float16* y);

  static void InitializeWithConstConf(DeviceCtx* ctx,
                                      const ConstantInitializerConf& initializer_conf, Blob* blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UTIL_CUDA_ARITHEMETIC_INTERFACE_H_
