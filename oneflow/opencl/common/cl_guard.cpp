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
#include "oneflow/opencl/common/cl_guard.h"

#include "oneflow/opencl/common/cl_util.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

clCurrentDeviceGuard::clCurrentDeviceGuard(int32_t dev_id) {
  CHECK(!pthread_fork::IsForkedSubProcess()) << pthread_fork::kOfDeviceNotSupportInForkedSubProcess;
  OF_CL_CHECK(clGetDevice(&saved_dev_id_));
  OF_CL_CHECK(clSetDevice(dev_id));
}

clCurrentDeviceGuard::clCurrentDeviceGuard() { OF_CL_CHECK(clGetDevice(&saved_dev_id_)); }

clCurrentDeviceGuard::~clCurrentDeviceGuard() { OF_CL_CHECK(clSetDevice(saved_dev_id_)); }

}  // namespace oneflow
