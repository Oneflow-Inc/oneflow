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
#include "oneflow/core/kernel/util/host_dnn_interface.h"

namespace oneflow {

namespace {

template<typename T>
static void ReluImpl(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  T zero = GetZeroVal<T>();
  for (int64_t i = 0; i != n; ++i) { y[i] = std::max(x[i], zero); }
}

template<typename T>
static void ReluBackwardImpl(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                             T* dx) {
  T zero = GetZeroVal<T>();
  for (int64_t i = 0; i != n; ++i) { dx[i] = (y[i] > zero) * dy[i]; }
}

template<typename T>
static void SigmoidImpl(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
  T half = static_cast<T>(0.5);
  for (int64_t i = 0; i != n; ++i) { y[i] = half * std::tanh(half * x[i]) + half; }
}

template<typename T>
static void SigmoidBackwardImpl(DeviceCtx* ctx, const int64_t n, const T* x, const T* y,
                                const T* dy, T* dx) {
  for (int64_t i = 0; i != n; ++i) { dx[i] = y[i] * (1 - y[i]) * dy[i]; }
}

}  // namespace

void DnnIf<DeviceType::kCPU>::Relu(DeviceCtx* ctx, const int64_t n, const float* x, float* y) {
  ReluImpl<float>(ctx, n, x, y);
}

void DnnIf<DeviceType::kCPU>::Relu(DeviceCtx* ctx, const int64_t n, const double* x, double* y) {
  ReluImpl<double>(ctx, n, x, y);
}

void DnnIf<DeviceType::kCPU>::ReluBackward(DeviceCtx* ctx, const int64_t n, const float* x,
                                           const float* y, const float* dy, float* dx) {
  ReluBackwardImpl<float>(ctx, n, x, y, dy, dx);
}

void DnnIf<DeviceType::kCPU>::ReluBackward(DeviceCtx* ctx, const int64_t n, const double* x,
                                           const double* y, const double* dy, double* dx) {
  ReluBackwardImpl<double>(ctx, n, x, y, dy, dx);
}

void DnnIf<DeviceType::kCPU>::Sigmoid(DeviceCtx* ctx, int64_t n, const float* x, float* y) {
  SigmoidImpl<float>(ctx, n, x, y);
}

void DnnIf<DeviceType::kCPU>::Sigmoid(DeviceCtx* ctx, int64_t n, const double* x, double* y) {
  SigmoidImpl<double>(ctx, n, x, y);
}

void DnnIf<DeviceType::kCPU>::SigmoidBackward(DeviceCtx* ctx, const int64_t n, const float* x,
                                              const float* y, const float* dy, float* dx) {
  SigmoidBackwardImpl<float>(ctx, n, x, y, dy, dx);
}

void DnnIf<DeviceType::kCPU>::SigmoidBackward(DeviceCtx* ctx, const int64_t n, const double* x,
                                              const double* y, const double* dy, double* dx) {
  SigmoidBackwardImpl<double>(ctx, n, x, y, dy, dx);
}

}  // namespace oneflow
