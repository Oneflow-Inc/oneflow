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
#include "oneflow/core/cuda/check_inf_nan.cuh"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void CheckInfNan(size_t elem_cnt, const T* elem, int* ret_code) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    bool is_inf = isinf(elem[i]);
    bool is_nan = isnan(elem[i]);
    if (is_inf) { *ret_code = 1; }
    if (is_nan) { *ret_code = 2; }
  }
}

}  // namespace

int CheckKernelOutputInfNan(DeviceCtx* ctx, Blob* blob) {
  if (!dynamic_cast<CudaDeviceCtx*>(ctx)) { return -1; }

  int* ret_code = nullptr;
  cudaMallocHost(&ret_code, sizeof(int));
  Memset<DeviceType::kCPU>(ctx, ret_code, 0, sizeof(int));

  size_t elem_cnt = blob->shape().elem_cnt();
  switch (blob->data_type()) {
    case DataType::kFloat: {
      RUN_CUDA_KERNEL((CheckInfNan<float>), ctx, elem_cnt, elem_cnt, blob->dptr<float>(),
                      ret_code);
      break;
    }
    default: {
      *ret_code = -2;
    }
  }

  int ret = *ret_code;
  cudaFreeHost(ret_code);
  return ret;
}

}  // namespace oneflow
