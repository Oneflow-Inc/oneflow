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

#include "oneflow/user/kernels/scalar_fmod_kernel.h"

namespace oneflow {

template<template<typename T> class BIN_OP, typename T>
__global__ void DoCUDAScalarFMod(const int64_t elem_cnt, const T scalar, const T* in, T* out) {
  DoScalarFmod<BIN_OP, T>(elem_cnt, scalar, in, out);
}

template<template<typename T> class BIN_OP, typename T>
struct ScalarFmodFunctor<DeviceType::kGPU, BIN_OP, T> final {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const T scalar, const T* in, T* out) {
    RUN_CUDA_KERNEL((DoCUDAScalarFMod<BIN_OP, T>), ctx, BlocksNum4ThreadsNum(elem_cnt), elem_cnt,
                    scalar, in, out);
  }
};

#define INSTANTIATE_SCALAR_FUNCTORS(device_type, binary_op)           \
  template struct ScalarFmodFunctor<device_type, binary_op, int8_t>;  \
  template struct ScalarFmodFunctor<device_type, binary_op, int32_t>; \
  template struct ScalarFmodFunctor<device_type, binary_op, int64_t>; \
  template struct ScalarFmodFunctor<device_type, binary_op, float>;   \
  template struct ScalarFmodFunctor<device_type, binary_op, double>;

INSTANTIATE_SCALAR_FUNCTORS(DeviceType::kGPU, BinaryFuncFMod);

}  // namespace oneflow
