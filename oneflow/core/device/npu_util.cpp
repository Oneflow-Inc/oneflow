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

#ifdef WITH_NPU
#include <mutex>
#include "oneflow/core/device/npu_util.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/hardware/node_device_descriptor_manager.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/ep/npu/npu_stream.h"
#include "oneflow/core/common/util.h"
#endif  // WITH_NPU

namespace oneflow {

#ifdef WITH_NPU


NpuCurrentDeviceGuard::NpuCurrentDeviceGuard(int32_t dev_id) {
  CHECK(!pthread_fork::IsForkedSubProcess()) << pthread_fork::kOfNpuNotSupportInForkedSubProcess;
  // if (!is_dev_init) 
  // {
  //   OF_NPU_CHECK(aclrtSetDevice(dev_id));
  //   is_dev_init = true;
  // }
  OF_NPU_CHECK(aclrtSetDevice(dev_id));
  OF_NPU_CHECK(aclrtGetDevice(&saved_dev_id_));
  // OF_NPU_CHECK(aclrtSetDevice(dev_id));
}

NpuCurrentDeviceGuard::NpuCurrentDeviceGuard() {
  OF_NPU_CHECK(aclrtGetDevice(&saved_dev_id_)); 
}

NpuCurrentDeviceGuard::~NpuCurrentDeviceGuard() { 
  OF_NPU_CHECK(aclrtSetDevice(saved_dev_id_));
}

int GetNpuDeviceIndex() { return GlobalProcessCtx::LocalRank(); }

int GetNpuDeviceCount() {
  std::cout<<"GetNpuDeviceCount"<<std::endl;
  /* static */ uint32_t npu_device_count = 0;
  NpuCurrentDeviceGuard dev_guard(GetNpuDeviceIndex());
  OF_NPU_CHECK(aclrtGetDeviceCount(&npu_device_count));
  return npu_device_count;
}

void InitNpuContextOnce(int device_id ) {
  std::cout<<"InitNpuContextOnce device_id"<<device_id<<std::endl;
  static std::once_flag aclcontext;
  static aclrtContext context_;
  std::call_once(aclcontext,[&](){
    std::cout<<"Init && Create Context Once"<<std::endl;
    OF_NPU_CHECK(aclInit(nullptr));
    OF_NPU_CHECK(aclrtCreateContext(&context_, device_id));
  });
  static int device_count = GetNpuDeviceCount();
  static std::vector<std::once_flag> init_flags = std::vector<std::once_flag>(device_count);
  if (LazyMode::is_enabled()) { return; }
  if (device_id == -1) { device_id = GetNpuDeviceIndex(); }
  std::call_once(init_flags[device_id], [&]() {
    OF_NPU_CHECK(aclrtSetDevice(device_id));
    OF_NPU_CHECK(aclrtSynchronizeDevice());
  });
}
#endif  // WITH_NPU

} // namespace oneflow