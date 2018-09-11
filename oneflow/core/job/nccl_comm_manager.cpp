#include <oneflow/core/control/ctrl_client.h>
#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

NcclCommMgr::NcclCommMgr(const Plan& plan) {
  std::map<int64_t, std::vector<TaskProto>> parallel_set2nccl_tasks;

  for (const auto& task : plan.task()) {
    if (task.machine_id() != Global<MachineCtx>::Get()->this_machine_id()) { continue; }
    if (!IsNcclTaskType(task.task_type())) { continue; }

    CHECK_EQ(Global<IDMgr>::Get()->GetDeviceTypeFromThrdId(task.thrd_id()), DeviceType::kGPU);
    CHECK(task.has_parallel_ctx());
    CHECK(task.parallel_ctx().has_parallel_set_id());

    parallel_set2nccl_tasks[task.parallel_ctx().parallel_set_id()].push_back(task);
  }

  for (const auto& pair : parallel_set2nccl_tasks) {

    std::vector<ncclComm_t> comms(pair.second.size());

    ncclUniqueId nccl_unique_id{};

    std::string nccl_unique_id_key = "nccl_unique_id_" + std::to_string(pair.first);

    if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
      NcclCheck(ncclGetUniqueId(&nccl_unique_id));
      Global<CtrlClient>::Get()->PushKV(nccl_unique_id_key,
                                        std::string(nccl_unique_id.internal, NCCL_UNIQUE_ID_BYTES));
    } else {
      Global<CtrlClient>::Get()->PullKV(
          nccl_unique_id_key, [&nccl_unique_id](const std::string& key) {
            memcpy(nccl_unique_id.internal, key.data(), NCCL_UNIQUE_ID_BYTES);
          });
    }

    ncclGroupStart();
    FOR_RANGE(size_t, i, 0, pair.second.size()) {
      const TaskProto& task = pair.second.at(i);
      int64_t task_id = task.task_id();
      int64_t thrd_id = Global<IDMgr>::Get()->ThrdId4ActorId(task_id);
      int32_t device_id = (int32_t)Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(thrd_id);
      cudaSetDevice(device_id);
      ncclCommInitRank(&comms[i], (int32_t)task.parallel_ctx().parallel_num(), nccl_unique_id,
                       (int32_t)task.parallel_ctx().parallel_id());
    }
    NcclCheck(ncclGroupEnd());

    FOR_RANGE(size_t, i, 0, pair.second.size()) {
      CHECK(actor_id2comm_.emplace(pair.second.at(i).task_id(), comms.at(i)).second);
    }
  }
}

NcclCommMgr::~NcclCommMgr() {
  for (const auto& pair : actor_id2comm_) { ncclCommDestroy(pair.second); }
}

ncclComm_t NcclCommMgr::NcclComm4ActorId(int64_t actor_id) const {
  auto it = actor_id2comm_.find(actor_id);
  if (it == actor_id2comm_.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

bool NcclCommMgr::IsNcclTaskType(const TaskType& tt) const {
  return tt == TaskType::kNcclAllGather || tt == TaskType::kNcclAllReduce
         || tt == TaskType::kNcclReduceScatter;
}

}  // namespace oneflow
