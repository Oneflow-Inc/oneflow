#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/job/machine_context.h"
#include "nccl_comm_manager.h"

namespace oneflow {
NcclCommMgr::NcclCommMgr(const Plan& plan) {
  HashMap<int64_t, std::vector<int64_t>> parallel_set2nccl_task_ids;

  for (const auto& task : plan.task()) {
    if (task.machine_id() != Global<MachineCtx>::Get()->this_machine_id()) { continue; }
    if (!IsNcclTaskType(task.task_type())) { continue; }

    CHECK_EQ(Global<IDMgr>::Get()->GetDeviceTypeFromThrdId(task.thrd_id()), DeviceType::kGPU);
    CHECK(task.has_parallel_ctx());
    CHECK(task.parallel_ctx().has_parallel_set_id());

    parallel_set2nccl_task_ids[task.parallel_ctx().parallel_set_id()].push_back(task.task_id());
  }

  for (const auto& pair : parallel_set2nccl_task_ids) {
    std::vector<ncclComm_t> comms(pair.second.size());
    std::vector<int> devices(pair.second.size());
    for (int i = 0; i < pair.second.size(); ++i) {
      int64_t thrd_id = Global<IDMgr>::Get()->ThrdId4ActorId(pair.second.at(i));
      devices[i] = (int)Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(thrd_id);
    }
    CudaCheck(ncclCommInitAll(comms.data(), (int)devices.size(), devices.data()));
    for (int i = 0; i < pair.second.size(); ++i) {
      CHECK(actor_id2comm_.emplace(pair.second.at(i), comms.at(i)).second);
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