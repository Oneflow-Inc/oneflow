/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_KERNEL_UTIL_HOST_DNN_INTERFACE_H_
#define ONEFLOW_CORE_KERNEL_UTIL_HOST_DNN_INTERFACE_H_

#include "oneflow/core/kernel/util/dnn_interface.h"

namespace oneflow {

template<>
struct DnnIf<DeviceType::kCPU> {
  static void Relu(DeviceCtx* ctx, const int64_t n, const float* x, float* y);
  static void Relu(DeviceCtx* ctx, const int64_t n, const double* x, double* y);
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                           const float* dy, float* dx);
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                           const double* dy, double* dx);
  static void Sigmoid(DeviceCtx* ctx, int64_t n, const float* x, float* y);
  static void Sigmoid(DeviceCtx* ctx, int64_t n, const double* x, double* y);
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                              const float* dy, float* dx);
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                              const double* dy, double* dx);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UTIL_HOST_DNN_INTERFACE_H_
