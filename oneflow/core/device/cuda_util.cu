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
#include <cub/cub.cuh>
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

int GetCudaSmVersion() {
  int sm_version, device_ordinal;
  OF_CUDA_CHECK(cudaGetDevice(&device_ordinal));
  OF_CUDA_CHECK(cub::SmVersion(sm_version, device_ordinal));
  return sm_version;
}

int GetCudaPtxVersion() {
  int ptx_version;
  OF_CUDA_CHECK(cub::PtxVersion(ptx_version));
  return ptx_version;
}

}  // namespace oneflow
