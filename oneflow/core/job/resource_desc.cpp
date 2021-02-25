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
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/common/util.h"
#ifdef WITH_CUDA
#include <nccl.h>
#endif

namespace oneflow {

size_t ResourceDesc::TotalMachineNum() const {
  CHECK_GT(resource_.machine_num(), 0);
  CHECK_LE(resource_.machine_num(), Global<EnvDesc>::Get()->TotalMachineNum());
  return resource_.machine_num();
}

const Machine& ResourceDesc::machine(int32_t idx) const {
  CHECK_LT(idx, TotalMachineNum());
  return Global<EnvDesc>::Get()->machine(idx);
}

int32_t ResourceDesc::ComputeThreadPoolSize() const {
  if (resource_.has_compute_thread_pool_size()) {
    CHECK_GT(resource_.compute_thread_pool_size(), 0);
    return resource_.compute_thread_pool_size();
  } else {
    return CpuDeviceNum();
  }
}

bool ResourceDesc::enable_debug_mode() const {
  return std::getenv("ONEFLOW_DEBUG_MODE") != nullptr || resource_.enable_debug_mode();
}

CollectiveBoxingConf ResourceDesc::collective_boxing_conf() const {
  if (resource_.has_collective_boxing_conf()) {
    return resource_.collective_boxing_conf();
  } else {
    return CollectiveBoxingConf();
  }
}

bool ResourceDesc::nccl_use_compute_stream() const {
#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700
  return resource_.nccl_use_compute_stream();
#else
  return false;
#endif
}

}  // namespace oneflow
