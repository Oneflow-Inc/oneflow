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
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/id_manager.h"

#ifdef WITH_CUDA

namespace oneflow {

namespace {

std::string GetNcclUniqueIdRpcKey(const std::vector<std::pair<int64_t, int64_t>>& sorted_devices) {
  std::ostringstream oss;
  oss << "eager_nccl_unique_id_rpc_key";
  for (const auto& pair : sorted_devices) { oss << ":" << pair.first << ":" << pair.second; }
  return oss.str();
}

}  // namespace

EagerNcclCommMgr::~EagerNcclCommMgr() {
  for (auto& device_set7device_id2comm : device_set2device_id2comm_) {
    for (auto& device_id7comm : device_set7device_id2comm.second) {
      OF_NCCL_CHECK(ncclCommDestroy(device_id7comm.second));
    }
  }
}

ncclComm_t EagerNcclCommMgr::GetCommForDevice(
    const std::set<std::pair<int64_t, int64_t>>& device_set) {
  int dev;
  OF_CUDA_CHECK(cudaGetDevice(&dev));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_set2device_id2comm_.find(device_set);
    if (it != device_set2device_id2comm_.end()) { return it->second.at(dev); }
  }
  std::vector<std::pair<int64_t, int64_t>> device_vec(device_set.cbegin(), device_set.cend());
  std::sort(device_vec.begin(), device_vec.end(),
            [](const std::pair<int64_t, int64_t>& a, const std::pair<int64_t, int64_t>& b) {
              if (a.first == b.first) {
                return a.second < b.second;
              } else {
                return a.first < b.first;
              }
            });
  int64_t machine = GlobalProcessCtx::Rank();
  std::pair<int64_t, int64_t> this_device(machine, dev);
  auto it = std::find(device_vec.cbegin(), device_vec.cend(), this_device);
  CHECK(it != device_vec.end());
  int rank = std::distance(device_vec.cbegin(), it);
  ncclUniqueId nccl_unique_id{};
  std::string nccl_unique_id_rpc_key = GetNcclUniqueIdRpcKey(device_vec);
  if (rank == 0) {
    OF_NCCL_CHECK(ncclGetUniqueId(&nccl_unique_id));
    Global<CtrlClient>::Get()->PushKV(nccl_unique_id_rpc_key,
                                      std::string(nccl_unique_id.internal, NCCL_UNIQUE_ID_BYTES));
  } else {
    Global<CtrlClient>::Get()->PullKV(
        nccl_unique_id_rpc_key, [&nccl_unique_id](const std::string& val) {
          memcpy(nccl_unique_id.internal, val.data(), NCCL_UNIQUE_ID_BYTES);
        });
  }
  ncclComm_t comm;
  OF_NCCL_CHECK(ncclCommInitRank(&comm, device_vec.size(), nccl_unique_id, rank));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    device_set2device_id2comm_[device_set][dev] = comm;
  }
  return comm;
}

}  // namespace oneflow

#endif  // WITH_CUDA
