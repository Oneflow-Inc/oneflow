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
#include "oneflow/cambricon/ep/mlu_util.h"

#include <mutex>
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/vm/vm_util.h"

namespace oneflow {

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
  CHECK(!pthread_fork::IsForkedSubProcess()) << pthread_fork::kOfDeviceNotSupportInForkedSubProcess;
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
  /* static */ uint32_t mlu_device_count = 0;
  MluCurrentDeviceGuard dev_guard(GetMluDeviceIndex());
  OF_MLU_CHECK(cnrtGetDeviceCount(&mlu_device_count));
  return mlu_device_count;
}

std::string cnnlErrorString(cnnlStatus_t status) {
  switch (status) {
    case CNNL_STATUS_SUCCESS: {
      return "CNNL_STATUS_SUCCESS";
    }
    case CNNL_STATUS_NOT_INITIALIZED: {
      return "CNNL_STATUS_NOT_INITIALIZED";
    }
    case CNNL_STATUS_ALLOC_FAILED: {
      return "CNNL_STATUS_ALLOC_FAILED";
    }
    case CNNL_STATUS_BAD_PARAM: {
      return "CNNL_STATUS_BAD_PARAM";
    }
    case CNNL_STATUS_INTERNAL_ERROR: {
      return "CNNL_STATUS_INTERNAL_ERROR";
    }
    case CNNL_STATUS_ARCH_MISMATCH: {
      return "CNNL_STATUS_MISMATCH";
    }
    case CNNL_STATUS_EXECUTION_FAILED: {
      return "CNNL_STATUS_EXECUTION_FAILED";
    }
    case CNNL_STATUS_NOT_SUPPORTED: {
      return "CNNL_STATUS_NOT_SUPPORTED";
    }
    case CNNL_STATUS_NUMERICAL_OVERFLOW: {
      return "CNNL_STATUS_NUMERICAL_OVERFLOW";
    }
    default: {
      return "CNNL_STATUS_UNKNOWN";
    }
  }
}

}  // namespace oneflow
