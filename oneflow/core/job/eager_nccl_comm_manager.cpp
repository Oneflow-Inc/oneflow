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
#include <string>
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/vm/vm_util.h"

#ifdef WITH_CUDA

namespace oneflow {

namespace {

std::string GetNcclUniqueIdRpcKey(const std::vector<std::pair<int64_t, int64_t>>& sorted_devices) {
  std::ostringstream oss;
  oss << "eager_nccl_unique_id_rpc_key";
  for (const auto& pair : sorted_devices) { oss << "," << pair.first << ":" << pair.second; }
  return oss.str();
}

std::string NcclUniqueId2String(const ncclUniqueId& id) {
  std::stringstream ss;
  for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
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

void CreateNcclComm(ncclComm_t* comm, const int dev, const std::string& key,
                    const std::vector<std::pair<int64_t, int64_t>>& device_vec) {
  ncclUniqueId nccl_unique_id{};
  int64_t machine = GlobalProcessCtx::Rank();
  std::pair<int64_t, int64_t> this_device(machine, dev);
  auto it = std::find(device_vec.cbegin(), device_vec.cend(), this_device);
  CHECK(it != device_vec.end());
  int rank = std::distance(device_vec.cbegin(), it);
  if (rank == 0) {
    OF_NCCL_CHECK(ncclGetUniqueId(&nccl_unique_id));
    Singleton<CtrlClient>::Get()->PushKV(
        key, std::string(nccl_unique_id.internal, NCCL_UNIQUE_ID_BYTES));
  } else {
    Singleton<CtrlClient>::Get()->PullKV(key, [&nccl_unique_id](const std::string& val) {
      memcpy(nccl_unique_id.internal, val.data(), NCCL_UNIQUE_ID_BYTES);
    });
  }
  VLOG(2) << " EagerNcclCommMgr::ncclCommInitRank device_vec.size() = " << device_vec.size()
          << ", nccl_unique_id = " << NcclUniqueId2String(nccl_unique_id) << ", rank = " << rank
          << ", key = {" << key << "}\n";
  OF_NCCL_CHECK(ncclCommInitRank(comm, device_vec.size(), nccl_unique_id, rank));
  VLOG(2) << " EagerNcclCommMgr::ncclCommInitRank succeed device_vec.size() = " << device_vec.size()
          << ", nccl_unique_id = " << NcclUniqueId2String(nccl_unique_id) << ", rank = " << rank
          << ", key = {" << key << "}\n";
}

bool NeedUnifiedNcclCommInit(const OperatorConf& op_conf) {
  if (op_conf.has_user_conf()) {
    return UserKernelUnifiedNcclCommInitRegistry::Instance().IsRegistered(
        op_conf.user_conf().op_type_name());
  } else {
    // Please check the .h file for hard-coding of the name
    return UserKernelUnifiedNcclCommInitRegistry::Instance().IsRegistered(
        kSystemOpPrefix + std::to_string(op_conf.op_type_case()));
  }
}

}  // namespace

const std::string EagerNcclCommMgr::kDefaultStreamName = "DEFAULT";

EagerNcclCommMgr::~EagerNcclCommMgr() {
  for (auto& device_set7device_id2comm : device_set2device_id2comm_) {
    for (auto& device_id7comm : device_set7device_id2comm.second) {
      OF_NCCL_CHECK(ncclCommDestroy(device_id7comm.second));
    }
  }
  for (auto& pair : device7stream2device_id2comm_) {
    for (auto& device_id7comm : pair.second) {
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
  std::sort(device_vec.begin(), device_vec.end(), CompareDeviceSetPair);

  ncclComm_t comm;
  std::string nccl_unique_id_rpc_key = GetNcclUniqueIdRpcKey(device_vec);
  CreateNcclComm(&comm, dev, nccl_unique_id_rpc_key, device_vec);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    device_set2device_id2comm_[device_set][dev] = comm;
  }
  return comm;
}

ncclComm_t EagerNcclCommMgr::GetCommForDeviceAndStreamName(
    const std::set<std::pair<int64_t, int64_t>>& device_set, const std::string& stream_name) {
  int dev;
  OF_CUDA_CHECK(cudaGetDevice(&dev));

  std::vector<std::pair<int64_t, int64_t>> device_vec(device_set.cbegin(), device_set.cend());
  std::sort(device_vec.begin(), device_vec.end(), CompareDeviceSetPair);
  std::string key = GetNcclUniqueIdRpcKey(device_vec) + "-stream_name_hint:" + stream_name;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device7stream2device_id2comm_.find(key);
    if (it != device7stream2device_id2comm_.end()) { return it->second.at(dev); }
  }

  ncclComm_t comm;
  CreateNcclComm(&comm, dev, key, device_vec);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    device7stream2device_id2comm_[key][dev] = comm;
  }
  return comm;
}

void EagerNcclCommMgr::CreateCommFromPlan(const Plan& plan) {
  const int64_t rank = GlobalProcessCtx::Rank();
  const int64_t dev = GlobalProcessCtx::LocalRank();
  std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> nccl_comm_key2devices;

  for (const auto& task_proto : plan.task()) {
    if (task_proto.machine_id() != rank) { continue; }
    if (task_proto.exec_sequence().exec_node_size() != 1) { continue; }
    const auto& kernel_conf = task_proto.exec_sequence().exec_node(0).kernel_conf();
    const OpAttribute* op_attr = nullptr;
    if (kernel_conf.has_op_attribute()) {
      op_attr = &kernel_conf.op_attribute();
    } else if (kernel_conf.has_op_attribute_ref()) {
      const auto& ref_name = kernel_conf.op_attribute_ref();
      op_attr = &plan.job_id2op_attribute_ref_table()
                     .at(task_proto.job_id())
                     .op_name2op_attribute()
                     .at(ref_name);
    } else {
      continue;
    }
    const auto& op_conf = op_attr->op_conf();
    if (!NeedUnifiedNcclCommInit(op_conf)) { continue; }
    if (!op_attr->has_parallel_conf_signature()) { continue; }
    if (!op_attr->parallel_conf_signature().has_op_parallel_conf()) { continue; }

    std::vector<std::pair<int64_t, int64_t>> device_vec;
    ParallelDesc parallel_desc(op_attr->parallel_conf_signature().op_parallel_conf());
    for (int64_t parallel_id = 0; parallel_id < parallel_desc.parallel_num(); ++parallel_id) {
      int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(parallel_id));
      int64_t device_id = CHECK_JUST(parallel_desc.DeviceId4ParallelId(parallel_id));
      device_vec.emplace_back(machine_id, device_id);
    }

    std::string stream_name = kDefaultStreamName;
    if (op_conf.has_stream_name_hint()) { stream_name = op_conf.stream_name_hint(); }
    std::string key = GetNcclUniqueIdRpcKey(device_vec) + "-stream_name_hint:" + stream_name;

    VLOG(3) << " EagerNcclCommMgr create nccl comm for " << op_conf.name() << ", rank = " << rank
            << ", dev = " << dev << ", key = {" << key << "}\n";
    nccl_comm_key2devices.emplace(std::move(key), std::move(device_vec));
  }

  if (nccl_comm_key2devices.size() == 0) { return; }

  CHECK_JUST(vm::CurrentRankSync());
  CudaCurrentDeviceGuard guard(dev);

  for (const auto& pair : nccl_comm_key2devices) {
    const auto& key = pair.first;
    auto device_id2comm_it = device7stream2device_id2comm_.find(key);
    if (device_id2comm_it != device7stream2device_id2comm_.end()) {
      auto comm_it = device_id2comm_it->second.find(dev);
      if (comm_it != device_id2comm_it->second.end()) { continue; }
    }
    ncclComm_t comm;
    CreateNcclComm(&comm, dev, key, pair.second);
    device7stream2device_id2comm_[key][dev] = comm;
  }
}

}  // namespace oneflow

#endif  // WITH_CUDA
