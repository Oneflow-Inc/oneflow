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
#include "oneflow/user/kernels/scalar_logical_kernels.h"
#include "oneflow/user/kernels/elementwise_xpu_kernel.cuh"

namespace oneflow {

template<template<typename T> class BIN_OP, typename T>
__global__ void DoCUDAScalarLogical(const int64_t elem_cnt, const T scalar, const T* in,
                                    int8_t* out) {
  DoScalarLogical<BIN_OP, T>(elem_cnt, scalar, in, out);
}

template<template<typename T> class BIN_OP, typename T>
struct ScalarLogicalFunctor<DeviceType::kGPU, BIN_OP, T> final {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const T scalar, const T* in,
                  int8_t* out) {
    RUN_CUDA_KERNEL((DoCUDAScalarLogical<BIN_OP, T>), ctx, BlocksNum4ThreadsNum(elem_cnt), elem_cnt,
                    scalar, in, out);
  }
};

INSTANTIATE_SCALAR_LOGICAL_FUNCTORS(DeviceType::kGPU, BinaryFuncEQ);
INSTANTIATE_SCALAR_LOGICAL_FUNCTORS(DeviceType::kGPU, BinaryFuncNE);
INSTANTIATE_SCALAR_LOGICAL_FUNCTORS(DeviceType::kGPU, BinaryFuncGT);
INSTANTIATE_SCALAR_LOGICAL_FUNCTORS(DeviceType::kGPU, BinaryFuncGE);
INSTANTIATE_SCALAR_LOGICAL_FUNCTORS(DeviceType::kGPU, BinaryFuncLT);
INSTANTIATE_SCALAR_LOGICAL_FUNCTORS(DeviceType::kGPU, BinaryFuncLE);
INSTANTIATE_SCALAR_LOGICAL_FUNCTORS(DeviceType::kGPU, BinaryFuncOR);
INSTANTIATE_SCALAR_LOGICAL_FUNCTORS(DeviceType::kGPU, BinaryFuncXOR);
INSTANTIATE_SCALAR_LOGICAL_FUNCTORS(DeviceType::kGPU, BinaryFuncAND);

}  // namespace oneflow
