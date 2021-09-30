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
#include "oneflow/core/kernel/cuda_check_device_kernel_observer.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

CudaCheckDeviceKernelObserver::CudaCheckDeviceKernelObserver(int device_id)
    : device_id_(device_id) {}

void CudaCheckDeviceKernelObserver::DidInit(KernelContext* kernel_ctx, const Kernel* kernel) {
#ifdef WITH_CUDA
  int device_id;
  OF_CUDA_CHECK(cudaGetDevice(&device_id));
  CHECK_EQ(device_id_, device_id) << kernel->op_conf().name() << " has set cuda device";
#endif  // WITH_CUDA
}

void CudaCheckDeviceKernelObserver::DidForward(KernelContext* kernel_ctx, const Kernel* kernel) {
#ifdef WITH_CUDA
  int device_id;
  OF_CUDA_CHECK(cudaGetDevice(&device_id));
  CHECK_EQ(device_id_, device_id) << kernel->op_conf().name() << " has set cuda device";
#endif  // WITH_CUDA
}

}  // namespace oneflow
