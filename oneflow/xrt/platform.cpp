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
#include "glog/logging.h"

#include "oneflow/xrt/platform.h"

#ifdef WITH_CUDA
#include "cuda_runtime.h"
#endif

namespace oneflow {
namespace xrt {

namespace platform {

int GetDeviceId(const XrtDevice &device) {
  switch (device) {
    case XrtDevice::CPU_X86: return 0;
    case XrtDevice::GPU_CUDA: {
#ifdef WITH_CUDA
      int device_id = 0;
      CHECK_EQ(cudaSuccess, cudaGetDevice(&device_id));
      return device_id;
#endif
    }
    case XrtDevice::GPU_CL:
    // TODO(hjchen2)
    case XrtDevice::CPU_ARM:
    // TODO(hjchen2)
    case XrtDevice::GPU_ARM:
      // TODO(hjchen2)
      return 0;
  }
  return 0;  // Compiler warning free
}

void SetDeviceId(const XrtDevice &device, const int device_id) {
  switch (device) {
    case XrtDevice::CPU_X86: return;
    case XrtDevice::GPU_CUDA: {
#ifdef WITH_CUDA
      CHECK_EQ(cudaSuccess, cudaSetDevice(device_id));
      return;
#endif
    }
    case XrtDevice::GPU_CL:
    // TODO(hjchen2)
    case XrtDevice::CPU_ARM:
    // TODO(hjchen2)
    case XrtDevice::GPU_ARM:
      // TODO(hjchen2)
      return;
  }
}

}  // namespace platform

}  // namespace xrt
}  // namespace oneflow
