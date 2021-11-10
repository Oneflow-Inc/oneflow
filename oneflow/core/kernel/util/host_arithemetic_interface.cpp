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
#include "oneflow/core/kernel/util/host_arithemetic_interface.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/operator/op_conf_util.h"

namespace oneflow {

namespace {

template<typename T>
void ConstantInitializer(const T& value, Blob* blob) {
  T* dptr = blob->mut_dptr<T>();
  const int64_t elem_cnt = blob->shape().elem_cnt();
  CHECK(elem_cnt);
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = value; }
}

}  // namespace

void ArithemeticIf<DeviceType::kCPU>::InitializeWithConstConf(
    DeviceCtx* ctx, const ConstantInitializerConf& initializer_conf, Blob* blob) {
  DataType dtype = blob->data_type();
  if (dtype == DataType::kFloat) {
    ConstantInitializer<float>(initializer_conf.value(), blob);
  } else if (dtype == DataType::kDouble) {
    ConstantInitializer<double>(static_cast<double>(initializer_conf.value()), blob);
  } else if (dtype == DataType::kFloat16) {
    ConstantInitializer<float16>(static_cast<float16>(initializer_conf.value()), blob);
  } else {
    UNIMPLEMENTED();
  }
}

#define MUL_BY_SCALAR(T)                                                                         \
  void ArithemeticIf<DeviceType::kCPU>::MulByScalar(DeviceCtx* ctx, const int64_t n, const T* x, \
                                                    const T y, T* z) {                           \
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] * y; }                                         \
  }

MUL_BY_SCALAR(float);
MUL_BY_SCALAR(double);
MUL_BY_SCALAR(int8_t);
MUL_BY_SCALAR(int32_t);
MUL_BY_SCALAR(int64_t);

#undef MUL_BY_SCALAR

#define ADD_BY_SCALAR(T)                                                                         \
  void ArithemeticIf<DeviceType::kCPU>::AddByScalar(DeviceCtx* ctx, const int64_t n, const T* x, \
                                                    const T y, T* z) {                           \
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] + y; }                                         \
  }

ADD_BY_SCALAR(float);
ADD_BY_SCALAR(double);
ADD_BY_SCALAR(int8_t);
ADD_BY_SCALAR(int32_t);
ADD_BY_SCALAR(int64_t);

#undef ADD_BY_SCALAR

#define MUL_BY_SCALAR_PTR(T)                                                            \
  void ArithemeticIf<DeviceType::kCPU>::MulByScalarPtr(DeviceCtx* ctx, const int64_t n, \
                                                       const T* x, const T* y, T* z) {  \
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] * y[0]; }                             \
  }

MUL_BY_SCALAR_PTR(float);
MUL_BY_SCALAR_PTR(double);
MUL_BY_SCALAR_PTR(int8_t);
MUL_BY_SCALAR_PTR(int32_t);
MUL_BY_SCALAR_PTR(int64_t);

#undef MUL_BY_SCALAR_PTR

#define ADD_BY_SCALAR_PTR(T)                                                            \
  void ArithemeticIf<DeviceType::kCPU>::AddByScalarPtr(DeviceCtx* ctx, const int64_t n, \
                                                       const T* x, const T* y, T* z) {  \
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] + y[0]; }                             \
  }

ADD_BY_SCALAR_PTR(float);
ADD_BY_SCALAR_PTR(double);
ADD_BY_SCALAR_PTR(int8_t);
ADD_BY_SCALAR_PTR(int32_t);
ADD_BY_SCALAR_PTR(int64_t);

#undef ADD_BY_SCALAR_PTR

#define SUB_BY_SCALAR_PTR(T)                                                            \
  void ArithemeticIf<DeviceType::kCPU>::SubByScalarPtr(DeviceCtx* ctx, const int64_t n, \
                                                       const T* x, const T* y, T* z) {  \
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] - y[0]; }                             \
  }

SUB_BY_SCALAR_PTR(float);
SUB_BY_SCALAR_PTR(double);
SUB_BY_SCALAR_PTR(int8_t);
SUB_BY_SCALAR_PTR(int32_t);
SUB_BY_SCALAR_PTR(int64_t);

#undef SUB_BY_SCALAR_PTR

#define DIV_BY_SCALAR_PTR(T)                                                            \
  void ArithemeticIf<DeviceType::kCPU>::DivByScalarPtr(DeviceCtx* ctx, const int64_t n, \
                                                       const T* x, const T* y, T* z) {  \
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] / y[0]; }                             \
  }

DIV_BY_SCALAR_PTR(float);
DIV_BY_SCALAR_PTR(double);
DIV_BY_SCALAR_PTR(int8_t);
DIV_BY_SCALAR_PTR(int32_t);
DIV_BY_SCALAR_PTR(int64_t);

#undef DIV_BY_SCALAR_PTR

}  // namespace oneflow
