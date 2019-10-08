#include "oneflow/core/job/in_job_mem_sharing_util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/id_manager.h"

namespace oneflow {

namespace {

int64_t GenDeviceUniqueId(int64_t machine_id, int64_t device_id) {
  return (machine_id << 32) | device_id;
}

HashMap<int64_t, std::vector<TaskProto*>> GenMemChainTasks(Plan* plan) {
  HashMap<int64_t, std::vector<TaskProto*>> device_unique_id2tasks;
  for (int64_t i = 0; i < plan->task_size(); ++i) {
    TaskProto* task = plan->mutable_task(i);
    int64_t machine_id = task->machine_id();
    DeviceType device_type = Global<IDMgr>::Get()->GetDeviceTypeFromThrdId(task->thrd_id());
    if (device_type == DeviceType::kCPU) { continue; }
    int64_t device_id = Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(task->thrd_id());
    int64_t device_unique_id = GenDeviceUniqueId(machine_id, device_id);
    device_unique_id2tasks[device_unique_id].push_back(task);
    /*
    for (auto& pair : *(task->mutable_produced_regst_desc())) {
      RegstDescProto* regst_desc = &pair.second;
    }
    */
  }
  return device_unique_id2tasks;
}

}  // namespace

void InJobMemSharingUtil::InferMemBlockId4MemReusedRegst(Plan* plan,
                                                         const PlanTaskGraph& plan_task_graph) {
  // 1 device 1 mem chain
  HashMap<int64_t, std::vector<TaskProto*>> mem_chain2tasks = GenMemChainTasks(plan);

  TODO();
}

}  // namespace oneflow
