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
#include <iomanip>
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/eager_hccl_comm_manager.h"
#include "oneflow/core/device/hccl_util.h"
#include "oneflow/core/job/id_manager.h"

#ifdef WITH_NPU

namespace oneflow {

namespace {

std::string GetHcclUniqueIdRpcKey(const std::vector<std::pair<int64_t, int64_t>>& sorted_devices) {
  std::ostringstream oss;
  oss << "eager_hccl_unique_id_rpc_key";
  for (const auto& pair : sorted_devices) { oss << "," << pair.first << ":" << pair.second; }
  return oss.str();
}

std::string HcclUniqueId2String(const HcclRootInfo& id) {
  std::stringstream ss;
  for (int i = 0; i < HCCL_ROOT_INFO_BYTES; ++i) {
    ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(id.internal[i]);
  }
  return ss.str();
}

bool CompareDeviceSetPair(const std::pair<int64_t, int64_t>& a,
                          const std::pair<int64_t, int64_t>& b) {
  if (a.first == b.first) {
    return a.second < b.second;
  } else {
    return a.first < b.first;
  }
}

void CreateHcclComm(HcclComm* comm, const int dev, const std::string& key,
                    const std::vector<std::pair<int64_t, int64_t>>& device_vec) {
  HcclRootInfo hccl_unique_id;
  int64_t machine = GlobalProcessCtx::Rank();
  std::pair<int64_t, int64_t> this_device(machine, dev);
  auto it = std::find(device_vec.cbegin(), device_vec.cend(), this_device);
  CHECK(it != device_vec.end());
  int rank = std::distance(device_vec.cbegin(), it);

  if (rank == 0) {
    OF_HCCL_CHECK(HcclGetRootInfo(&hccl_unique_id));
    Singleton<CtrlClient>::Get()->PushKV(key,
                                      std::string(hccl_unique_id.internal, HCCL_ROOT_INFO_BYTES));
  } else {
    Singleton<CtrlClient>::Get()->PullKV(key, [&hccl_unique_id](const std::string& val) {
      memcpy(hccl_unique_id.internal, val.data(), HCCL_ROOT_INFO_BYTES);
    });
  }
  VLOG(2) << " EagerHcclCommMgr::hcclCommInitRank device_vec.size() = " << device_vec.size()
          << ", hccl_unique_id = " << HcclUniqueId2String(hccl_unique_id) << ", rank = " << rank
          << ", key = {" << key << "}\n";

  OF_HCCL_CHECK(HcclCommInitRootInfo(device_vec.size(),&hccl_unique_id,rank,comm));

}

}  // namespace

EagerHcclCommMgr::~EagerHcclCommMgr() {
  for (auto& device_set7device_id2comm : device_set2device_id2comm_) {
    for (auto& device_id7comm : device_set7device_id2comm.second) {
      OF_HCCL_CHECK(HcclCommDestroy(device_id7comm.second));
    }
  }
  for (auto& pair : device7stream2device_id2comm_) {
    for (auto& device_id7comm : pair.second) {
      OF_HCCL_CHECK(HcclCommDestroy(device_id7comm.second));
    }
  }
}

HcclComm EagerHcclCommMgr::GetCommForDevice(
    const std::set<std::pair<int64_t, int64_t>>& device_set) {
  int dev;
  OF_NPU_CHECK(aclrtGetDevice(&dev));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_set2device_id2comm_.find(device_set);
    if (it != device_set2device_id2comm_.end()) { return it->second.at(dev); }
  }
  std::vector<std::pair<int64_t, int64_t>> device_vec(device_set.cbegin(), device_set.cend());
  std::sort(device_vec.begin(), device_vec.end(), CompareDeviceSetPair);

  HcclComm comm;
  std::string hccl_unique_id_rpc_key = GetHcclUniqueIdRpcKey(device_vec);
  CreateHcclComm(&comm, dev, hccl_unique_id_rpc_key, device_vec);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    device_set2device_id2comm_[device_set][dev] = comm;
  }
  return comm;
}

HcclComm EagerHcclCommMgr::GetCommForDeviceAndStreamName(
    const std::set<std::pair<int64_t, int64_t>>& device_set, const std::string& stream_name) {
  int dev;
  OF_NPU_CHECK(aclrtGetDevice(&dev));

  std::vector<std::pair<int64_t, int64_t>> device_vec(device_set.cbegin(), device_set.cend());
  std::sort(device_vec.begin(), device_vec.end(), CompareDeviceSetPair);
  std::string key = GetHcclUniqueIdRpcKey(device_vec) + "-stream_name_hint:" + stream_name;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device7stream2device_id2comm_.find(key);
    if (it != device7stream2device_id2comm_.end()) { return it->second.at(dev); }
  }

  HcclComm comm;
  CreateHcclComm(&comm, dev, key, device_vec);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    device7stream2device_id2comm_[key][dev] = comm;
  }
  return comm;
}

}  // namespace oneflow

#endif  // WITH_NPU
