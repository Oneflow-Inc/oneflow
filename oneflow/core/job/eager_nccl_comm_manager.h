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
#ifndef ONEFLOW_CORE_JOB_EAGER_NCCL_COMM_MANAGER_H_
#define ONEFLOW_CORE_JOB_EAGER_NCCL_COMM_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"

#ifdef WITH_CUDA

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

class EagerNcclCommMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerNcclCommMgr);
  ~EagerNcclCommMgr();

  ncclComm_t GetCommForDevice(const std::set<std::pair<int64_t, int64_t>>& device_set);

 private:
  friend class Global<EagerNcclCommMgr>;
  EagerNcclCommMgr() = default;

  std::map<std::set<std::pair<int64_t, int64_t>>, HashMap<int64_t, ncclComm_t>>
      device_set2device_id2comm_;
  std::mutex mutex_;
};

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_JOB_EAGER_NCCL_COMM_MANAGER_H_
