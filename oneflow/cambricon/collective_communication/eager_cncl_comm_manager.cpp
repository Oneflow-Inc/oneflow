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
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/cambricon/collective_communication/eager_cncl_comm_manager.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/common/mlu_guard.h"
#include "oneflow/core/graph/boxing/collective_boxing.pb.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {

namespace {

static const int64_t kNumOfCommInCurProcess = 1;

std::string GetcnclCliqueIdRpcKey(const std::vector<std::pair<int64_t, int64_t>>& sorted_devices) {
  std::ostringstream oss;
  oss << "eager_cncl_unique_id_rpc_key";
  for (const auto& pair : sorted_devices) { oss << "," << pair.first << ":" << pair.second; }
  return oss.str();
}

std::string cnclCliqueId2String(const cnclCliqueId& id) {
  std::stringstream ss;
  for (int i = 0; i < CNCL_CLIQUE_ID_BYTES_SIZE; ++i) {
    ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(id.data[i]);
  }
  ss << id.hash;
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

void CreateCnclComm(cnclComm_t* comm, const int dev, const std::string& key,
                    const std::vector<std::pair<int64_t, int64_t>>& device_vec) {
  cnclCliqueId cncl_unique_id{};
  int64_t machine = GlobalProcessCtx::Rank();
  std::pair<int64_t, int64_t> this_device(machine, dev);
  auto it = std::find(device_vec.cbegin(), device_vec.cend(), this_device);
  CHECK(it != device_vec.end());
  int rank = std::distance(device_vec.cbegin(), it);
  if (rank == 0) {
    OF_CNCL_CHECK(cnclGetCliqueId(&cncl_unique_id));
    Singleton<CtrlClient>::Get()->PushKV(key, CnclCliqueIdToString(cncl_unique_id));
  } else {
    Singleton<CtrlClient>::Get()->PullKV(key, [&cncl_unique_id](const std::string& val) {
      CnclCliqueIdFromString(val, &cncl_unique_id);
    });
  }
  int32_t dev_list[kNumOfCommInCurProcess] = {dev};
  int32_t rank_list[kNumOfCommInCurProcess] = {rank};
  VLOG(2) << " EagerCnclCommMgr::cnclCommInitRank device_vec.size() = " << device_vec.size()
          << ", cncl_unique_id = " << cnclCliqueId2String(cncl_unique_id) << ", rank = " << rank
          << ", key = {" << key << "}\n";
  OF_CNCL_CHECK(cnclInitComms(comm, kNumOfCommInCurProcess, dev_list, rank_list, device_vec.size(),
                              &cncl_unique_id));
  VLOG(2) << " EagerCnclCommMgr::cnclCommInitRank succeed device_vec.size() = " << device_vec.size()
          << ", cncl_unique_id = " << cnclCliqueId2String(cncl_unique_id) << ", rank = " << rank
          << ", key = {" << key << "}\n";
}

bool NeedUnifiedCnclCommInit(const OperatorConf& op_conf) {
  if (op_conf.has_user_conf()) {
    return UserKernelUnifiedCnclCommInitRegistry::Instance().IsRegistered(
        op_conf.user_conf().op_type_name());
  } else {
    // Please check the .h file for hard-coding of the name
    return UserKernelUnifiedCnclCommInitRegistry::Instance().IsRegistered(
        kSystemOpPrefix + std::to_string(op_conf.op_type_case()));
  }
}

}  // namespace

const std::string EagerCnclCommMgr::kDefaultStreamName = "DEFAULT";

EagerCnclCommMgr::~EagerCnclCommMgr() {
  for (auto& device_set7device_id2comm : device_set2device_id2comm_) {
    for (auto& device_id7comm : device_set7device_id2comm.second) {
      OF_CNCL_CHECK(cnclFreeComm(device_id7comm.second));
    }
  }
  for (auto& pair : device7stream2device_id2comm_) {
    for (auto& device_id7comm : pair.second) { OF_CNCL_CHECK(cnclFreeComm(device_id7comm.second)); }
  }
}

cnclComm_t EagerCnclCommMgr::GetCommForDevice(
    const std::set<std::pair<int64_t, int64_t>>& device_set) {
  int dev;
  CNRT_CHECK(cnrtGetDevice(&dev));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_set2device_id2comm_.find(device_set);
    if (it != device_set2device_id2comm_.end()) { return it->second.at(dev); }
  }
  std::vector<std::pair<int64_t, int64_t>> device_vec(device_set.cbegin(), device_set.cend());
  std::sort(device_vec.begin(), device_vec.end(), CompareDeviceSetPair);

  cnclComm_t comm;
  std::string cncl_unique_id_rpc_key = GetcnclCliqueIdRpcKey(device_vec);
  CreateCnclComm(&comm, dev, cncl_unique_id_rpc_key, device_vec);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    device_set2device_id2comm_[device_set][dev] = comm;
  }
  return comm;
}

cnclComm_t EagerCnclCommMgr::GetCommForDeviceAndStreamName(
    const std::set<std::pair<int64_t, int64_t>>& device_set, const std::string& stream_name) {
  int dev;
  CNRT_CHECK(cnrtGetDevice(&dev));

  std::vector<std::pair<int64_t, int64_t>> device_vec(device_set.cbegin(), device_set.cend());
  std::sort(device_vec.begin(), device_vec.end(), CompareDeviceSetPair);
  std::string key = GetcnclCliqueIdRpcKey(device_vec) + "-stream_name_hint:" + stream_name;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device7stream2device_id2comm_.find(key);
    if (it != device7stream2device_id2comm_.end()) { return it->second.at(dev); }
  }

  cnclComm_t comm;
  CreateCnclComm(&comm, dev, key, device_vec);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    device7stream2device_id2comm_[key][dev] = comm;
  }
  return comm;
}

void EagerCnclCommMgr::CreateCommFromPlan(const Plan& plan) {
  const int64_t rank = GlobalProcessCtx::Rank();
  const int64_t dev = GlobalProcessCtx::LocalRank();
  std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> cncl_comm_key2devices;

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
    if (!NeedUnifiedCnclCommInit(op_conf)) { continue; }
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
    std::string key = GetcnclCliqueIdRpcKey(device_vec) + "-stream_name_hint:" + stream_name;

    VLOG(3) << " EagerCnclCommMgr create cncl comm for " << op_conf.name() << ", rank = " << rank
            << ", dev = " << dev << ", key = {" << key << "}\n";
    cncl_comm_key2devices.emplace(std::move(key), std::move(device_vec));
  }

  if (cncl_comm_key2devices.size() == 0) { return; }

  CHECK_JUST(vm::CurrentRankSync());
  MluCurrentDeviceGuard guard(dev);

  for (const auto& pair : cncl_comm_key2devices) {
    const auto& key = pair.first;
    auto device_id2comm_it = device7stream2device_id2comm_.find(key);
    if (device_id2comm_it != device7stream2device_id2comm_.end()) {
      auto comm_it = device_id2comm_it->second.find(dev);
      if (comm_it != device_id2comm_it->second.end()) { continue; }
    }
    cnclComm_t comm;
    CreateCnclComm(&comm, dev, key, pair.second);
    device7stream2device_id2comm_[key][dev] = comm;
  }
}

COMMAND({
  CHECK(setenv("CNCL_LOG_LEVEL", "ERROR", 0) == 0);
  CHECK(setenv("CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE", "6", 0) == 0);
});

REGISTER_CCL_COMM_MGR(DeviceType::kMLU, EagerCnclCommMgr);

}  // namespace oneflow
