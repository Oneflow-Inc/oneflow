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

#include "oneflow/core/ep/cuda/cuda_matmul_mode.h"

namespace oneflow {

namespace ep {

namespace {

bool* GetThreadLocalMatmulAllowTF32() {
  static thread_local bool matmul_allow_tf32 = false;
  return &matmul_allow_tf32;
}

bool* GetThreadLocalMatmulAllowHalfPrecisionAccumulation() {
  static thread_local bool matmul_allow_half_precision_accumulation = false;
  return &matmul_allow_half_precision_accumulation;
}

}  // namespace

bool CudaMatmulMode::is_matmul_allow_tf32() { return *GetThreadLocalMatmulAllowTF32(); }

void CudaMatmulMode::set_matmul_allow_tf32(bool matmul_allow_tf32) {
  *GetThreadLocalMatmulAllowTF32() = matmul_allow_tf32;
}

bool CudaMatmulMode::is_matmul_allow_half_precision_accumulation() {
  return *GetThreadLocalMatmulAllowHalfPrecisionAccumulation();
}

void CudaMatmulMode::set_matmul_allow_half_precision_accumulation(
    bool matmul_allow_half_precision_accumulation) {
  *GetThreadLocalMatmulAllowHalfPrecisionAccumulation() = matmul_allow_half_precision_accumulation;
}

}  // namespace ep

}  // namespace oneflow
