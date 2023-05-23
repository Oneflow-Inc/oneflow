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

bool* GetThreadLocalMatmulAllowFP16ReducedPrecisionReducton() {
  static thread_local bool matmul_allow_fp16_reduced_precision_reduction = true;
  return &matmul_allow_fp16_reduced_precision_reduction;
}

}  // namespace

bool CudaMatmulMode::is_matmul_allow_tf32() { return *GetThreadLocalMatmulAllowTF32(); }

void CudaMatmulMode::set_matmul_allow_tf32(bool matmul_allow_tf32) {
  *GetThreadLocalMatmulAllowTF32() = matmul_allow_tf32;
}

bool CudaMatmulMode::is_matmul_allow_fp16_reduced_precision_reduction() {
  return *GetThreadLocalMatmulAllowFP16ReducedPrecisionReducton();
}

void CudaMatmulMode::set_matmul_allow_fp16_reduced_precision_reduction(
    bool matmul_allow_fp16_reduced_precision_reduction) {
  *GetThreadLocalMatmulAllowFP16ReducedPrecisionReducton() =
      matmul_allow_fp16_reduced_precision_reduction;
}

}  // namespace ep

}  // namespace oneflow
