#include "oneflow/core/schedule/plan_sgraph.h"
#include "oneflow/core/job/id_manager.h"
namespace oneflow {
namespace schedule {

void PlanSGraph::InitFromPlan(const Plan& plan) {
  for (const TaskProto& task_proto : plan.task()) {
    uint64_t task_id = task_proto.id();
    //	STask
    STask* node = mut_node_mgr().CreateWithId(task_id);
    if (node) { mut_children_arc_mgr().CreateIfNotFound(this, node); }
    bool is_copy_hd = (task_proto.type() == kCopyHdTask);
    //	SDevice
    uint64_t device_id =
        IDMgr::Singleton()->UUDeviceId4TaskId(task_proto.id(), is_copy_hd);
    std::string device_name = std::to_string(device_id);
    SDevice* device = mut_device_mgr().CreateIfNotFound(device_id);
    device->mut_time() = 1;
    mut_device_arc_mgr().CreateIfNotFound(node, device);
  }

  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      uint64_t regst_desc_id = pair.second.regst_desc_id();
      uint64_t producer_id = pair.second.producer_task_id();
      STask* from = mut_node_mgr().Find(producer_id);
      SRegstDesc* to = mut_regst_desc_mgr().CreateIfNotFound(regst_desc_id);
      if (from && to) {
        mut_produced_regst_desc_mgr().CreateIfNotFound(from, to);
      }
      for (int64_t consumer_id : pair.second.consumer_task_id()) {
        STask* from = mut_node_mgr().Find(consumer_id);
        if (from && to) {
          mut_subscribed_regst_desc_mgr().CreateIfNotFound(from, to);
        }
      }
    }
  }
}

void PlanSGraph::InitDevice() {}

}  // namespace schedule
}  // namespace oneflow
