#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/device/nccl_util.h"

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
    std::vector<std::pair<int64_t, int32_t>> task_id_device_id(pair.second.size());
    for (size_t i = 0; i < pair.second.size(); ++i) {
      int64_t task_id = pair.second.at(i);
      int64_t thrd_id = Global<IDMgr>::Get()->ThrdId4ActorId(task_id);
      int32_t device_id = (int32_t)Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(thrd_id);
      task_id_device_id[i] = {task_id, device_id};
    }

    std::sort(task_id_device_id.begin(), task_id_device_id.end(),
              [](const std::pair<int64_t, int32_t>& a, const std::pair<int64_t, int32_t>& b) {
                return a.first < b.first;
              });

    std::vector<ncclComm_t> comms(task_id_device_id.size());
    std::vector<int32_t> devices(task_id_device_id.size());
    for (size_t i = 0; i < task_id_device_id.size(); ++i) {
      devices[i] = task_id_device_id.at(i).second;
    }
    NcclCheck(ncclCommInitAll(comms.data(), (int32_t)devices.size(), devices.data()));
    for (size_t i = 0; i < task_id_device_id.size(); ++i) {
      CHECK(actor_id2comm_.emplace(task_id_device_id.at(i).first, comms.at(i)).second);
      int32_t device;
      int32_t rank;
      ncclCommCuDevice(comms.at(i), &device);
      ncclCommUserRank(comms.at(i), &rank);
      LOG(INFO) << "Created nccl communicator for task " << task_id_device_id.at(i).first
                << " with rank " << rank << " on device " << device;
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
