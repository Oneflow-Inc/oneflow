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
#include <mutex>
#include "oneflow/core/device/gcu_util.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/vm/vm_util.h"


namespace oneflow {

#ifdef WITH_MLU

namespace {

std::function<cnrtRet_t(void**, size_t)> GetMluMallocHostFn(int32_t dev) {
  auto default_fn = [](void** ptr, size_t size) { return cnrtHostMalloc(ptr, size); };
  return default_fn;
}

}  // namespace

cnrtRet_t NumaAwareMluMallocHost(int32_t dev, void** ptr, size_t size) {
  auto fn = GetMluMallocHostFn(dev);
  return fn(ptr, size);
}

MluCurrentDeviceGuard::MluCurrentDeviceGuard(int32_t dev_id) {
  CHECK(!pthread_fork::IsForkedSubProcess()) << pthread_fork::kOfMluNotSupportInForkedSubProcess;
  OF_MLU_CHECK(cnrtGetDevice(&saved_dev_id_));
  OF_MLU_CHECK(cnrtSetDevice(dev_id));
}

MluCurrentDeviceGuard::MluCurrentDeviceGuard() { OF_MLU_CHECK(cnrtGetDevice(&saved_dev_id_)); }

MluCurrentDeviceGuard::~MluCurrentDeviceGuard() { OF_MLU_CHECK(cnrtSetDevice(saved_dev_id_)); }

void MluSynchronize(int device_id) {
  MluCurrentDeviceGuard dev_guard(device_id);
  OF_MLU_CHECK(cnrtSyncDevice());
}

void SetMluDeviceIndex(int device_id) { OF_MLU_CHECK(cnrtSetDevice(device_id)); }

int GetMluDeviceIndex() { return GlobalProcessCtx::LocalRank(); }

int GetMluDeviceCount() {
  /* static */ int gcu_device_count = 0;
  MluCurrentDeviceGuard dev_guard(GetMluDeviceIndex());
  OF_MLU_CHECK(cnrtGetDeviceCount(&gcu_device_count));
  return gcu_device_count;
}

static std::once_flag prop_init_flag;

void InitMluContextOnce(int device_id) {
  static int device_count = GetMluDeviceCount();
  static std::vector<std::once_flag> init_flags = std::vector<std::once_flag>(device_count);
  if (LazyMode::is_enabled()) { return; }
  if (device_id == -1) { device_id = GetMluDeviceIndex(); }
  std::call_once(init_flags[device_id], [&]() {
    OF_MLU_CHECK(cnrtSetDevice(device_id));
    OF_MLU_CHECK(cnrtSyncDevice());
  });
}

#endif  // WITH_MLU

}  // namespace oneflow
