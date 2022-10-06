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
#ifndef ONEFLOW_CORE_JOB_EAGER_HCCL_COMM_MANAGER_H_
#define ONEFLOW_CORE_JOB_EAGER_HCCL_COMM_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"

#ifdef WITH_NPU

#include "oneflow/core/device/npu_util.h"
#include "hccl/hccl.h"
namespace oneflow {

class EagerHcclCommMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerHcclCommMgr);
  ~EagerHcclCommMgr();

  HcclComm GetCommForDevice(const std::set<std::pair<int64_t, int64_t>>& device_set);
  HcclComm GetCommForDeviceAndStreamName(const std::set<std::pair<int64_t, int64_t>>& device_set,
                                           const std::string& stream_name);

 private:
  friend class Singleton<EagerHcclCommMgr>;
  EagerHcclCommMgr() = default;

  std::map<std::set<std::pair<int64_t, int64_t>>, HashMap<int64_t, HcclComm>>
      device_set2device_id2comm_;
  std::map<std::string, HashMap<int64_t, HcclComm>> device7stream2device_id2comm_;
  std::mutex mutex_;
};

}  // namespace oneflow

#endif  // WITH_NPU

#endif  // ONEFLOW_CORE_JOB_EAGER_HCCL_COMM_MANAGER_H_
