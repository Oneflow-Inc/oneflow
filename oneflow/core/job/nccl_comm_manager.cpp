#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/device/nccl_util.h"

#ifdef WITH_CUDA

namespace oneflow {

NcclCommMgr::NcclCommMgr(const Plan& plan) {
  std::map<int64_t, std::vector<TaskProto>> rank_set2nccl_tasks;

  for (const auto& task : plan.task()) {
    if (task.machine_id() != Global<MachineCtx>::Get()->this_machine_id()) { continue; }
    if (!IsNcclTaskType(task.task_type())) { continue; }

    CHECK_EQ(Global<IDMgr>::Get()->GetDeviceTypeFromThrdId(task.thrd_id()), DeviceType::kGPU);
    CHECK(task.has_parallel_ctx());
    CHECK(task.parallel_ctx().has_rank_ctx());

    if (!rank_set2nccl_tasks[task.parallel_ctx().rank_ctx().rank_set_id()].empty()) {
      TaskProto& first = rank_set2nccl_tasks[task.parallel_ctx().rank_ctx().rank_set_id()].front();
      CHECK_EQ(first.task_type(), task.task_type());
      CHECK_EQ(first.parallel_ctx().rank_ctx().rank_num(),
               task.parallel_ctx().rank_ctx().rank_num());
    }

    rank_set2nccl_tasks[task.parallel_ctx().rank_ctx().rank_set_id()].push_back(task);
  }

  for (const auto& pair : rank_set2nccl_tasks) {
    ncclUniqueId nccl_unique_id{};
    NcclGetUniqueId4Tasks(pair.second, &nccl_unique_id);
    std::vector<ncclComm_t> comms(pair.second.size());
    NcclCommInitRank4Tasks(pair.second, &comms, nccl_unique_id);
    FOR_RANGE(size_t, i, 0, pair.second.size()) {
      int32_t device_id;
      int32_t rank;
      NcclCheck(ncclCommCuDevice(comms.at(i), &device_id));
      NcclCheck(ncclCommUserRank(comms.at(i), &rank));
      LOG(INFO) << "Created nccl communicator for task " << pair.second.at(i).task_id()
                << " with rank " << rank << " on device " << device_id;
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
         || tt == TaskType::kNcclReduceScatter || tt == TaskType::kNcclBoxingAllGather
         || tt == TaskType::kNcclBoxingAllReduce || tt == TaskType::kNcclBoxingReduceScatter
         || tt == TaskType::kNcclReduceScatter || tt == TaskType::kNcclTupleBroadcast
         || tt == TaskType::kNcclTupleReduce;
}

int32_t NcclCommMgr::GetDeviceId4Task(const TaskProto& task) {
  int64_t thrd_id = Global<IDMgr>::Get()->ThrdId4ActorId(task.task_id());
  return (int32_t)Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(thrd_id);
}

void NcclCommMgr::NcclCommInitRank4Tasks(const std::vector<TaskProto>& tasks,
                                         std::vector<ncclComm_t>* comms,
                                         ncclUniqueId nccl_unique_id) {
  NcclCheck(ncclGroupStart());
  FOR_RANGE(size_t, i, 0, tasks.size()) {
    CudaCurrentDeviceGuard guard(GetDeviceId4Task(tasks.at(i)));
    NcclCheck(
        ncclCommInitRank(&((*comms)[i]), (int32_t)tasks.at(i).parallel_ctx().rank_ctx().rank_num(),
                         nccl_unique_id, (int32_t)tasks.at(i).parallel_ctx().rank_ctx().rank_id()));
  }
  NcclCheck(ncclGroupEnd());
}

void NcclCommMgr::NcclGetUniqueId4Tasks(const std::vector<TaskProto>& tasks,
                                        ncclUniqueId* nccl_unique_id) {
  if (tasks.size() == tasks.front().parallel_ctx().rank_ctx().rank_num()) {
    NcclCheck(ncclGetUniqueId(nccl_unique_id));
  } else {
    bool should_create_unique_id =
        std::find_if(
            tasks.begin(), tasks.end(),
            [](const TaskProto& task) { return task.parallel_ctx().rank_ctx().rank_id() == 0; })
        != tasks.end();
    std::string nccl_unique_id_rpc_key =
        "nccl_unique_id_" + std::to_string(tasks.front().parallel_ctx().rank_ctx().rank_set_id());
    if (should_create_unique_id) {
      NcclCheck(ncclGetUniqueId(nccl_unique_id));
      Global<CtrlClient>::Get()->PushKV(
          nccl_unique_id_rpc_key, std::string(nccl_unique_id->internal, NCCL_UNIQUE_ID_BYTES));
    } else {
      Global<CtrlClient>::Get()->PullKV(
          nccl_unique_id_rpc_key, [&nccl_unique_id](const std::string& val) {
            memcpy(nccl_unique_id->internal, val.data(), NCCL_UNIQUE_ID_BYTES);
          });
    }
  }
}

}  // namespace oneflow

#endif  // WITH_CUDA
