#ifndef ONEFLOW_CORE_KERNEL_UTIL_CUDA_DNN_INTERFACE_H_
#define ONEFLOW_CORE_KERNEL_UTIL_CUDA_DNN_INTERFACE_H_

#include "oneflow/core/kernel/util/dnn_interface.h"

namespace oneflow {

template<>
class DnnIf<DeviceType::kGPU> {
  static void Relu(DeviceCtx* ctx, const int64_t n, const float* x, float* y);
  static void Relu(DeviceCtx* ctx, const int64_t n, const double* x, double* y);
  static void Relu(DeviceCtx* ctx, const int64_t n, const float16* x, float16* y);
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                           const float* dy, float* dx);
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                           const double* dy, double* dx);
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y,
                           const float16* dy, float16* dx);
  static void Sigmoid(DeviceCtx* ctx, int64_t n, const float* x, float* y);
  static void Sigmoid(DeviceCtx* ctx, int64_t n, const double* x, double* y);
  static void Sigmoid(DeviceCtx* ctx, int64_t n, const float16* x, float16* y);
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                              const float* dy, float* dx);
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                              const double* dy, double* dx);
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y,
                              const float16* dy, float16* dx);
  static void TanH(DeviceCtx* ctx, int64_t n, const float* x, float* y);
  static void TanH(DeviceCtx* ctx, int64_t n, const double* x, double* y);
  static void TanH(DeviceCtx* ctx, int64_t n, const float16* x, float16* y);
  static void TanHBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                           const float* dy, float* dx);
  static void TanHBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                           const double* dy, double* dx);
  static void TanHBackward(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y,
                           const float16* dy, float16* dx);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UTIL_CUDA_DNN_INTERFACE_H_
