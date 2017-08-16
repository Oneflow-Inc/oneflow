#include "oneflow/core/schedule/plan_sgraph.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/operator/operator_manager.h"
namespace oneflow {
namespace schedule {

void PlanSGraph::InitRegstDesc(const Plan& plan) {
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      uint64_t regst_desc_id = pair.second.regst_desc_id();
      uint64_t producer_id = pair.second.producer_task_id();
      STask* producer = mut_node_mgr().Find(producer_id);
      SRegstDesc* regst_desc =
          mut_regst_desc_mgr().CreateIfNotFound(regst_desc_id);
      if (producer) {
        mut_produced_regst_desc_mgr().CreateIfNotFound(producer, regst_desc);
      }
      for (int64_t consumer_id : pair.second.consumer_task_id()) {
        STask* consumer = mut_node_mgr().Find(consumer_id);
        if (consumer) {
          mut_subscribed_regst_desc_mgr().CreateIfNotFound(consumer,
                                                           regst_desc);
        }
        if (consumer && producer) {
          mut_arc_mgr().CreateIfNotFound(producer, consumer);
        }
      }
    }
  }
}

void PlanSGraph::InitTask(const Plan& plan) {
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
}

void PlanSGraph::InitLoss(const Plan& plan) {
  std::unordered_map<int64_t, bool> task_id2is_loss;
  GenerateTaskId2IsLoss(plan, &task_id2is_loss);
  for (const TaskProto& task_proto : plan.task()) {
    if (task_id2is_loss[task_proto.id()]) {
      STask* loss = node_mgr().Find(task_proto.id());
      CHECK(loss);
      mut_loss_arc_mgr().CreateIfNotFound(this, loss);
    }
  }
}

void PlanSGraph::InitFromPlan(const Plan& plan) {
  InitTask(plan);
  InitLoss(plan);
  InitRegstDesc(plan);
}

void PlanSGraph::GenerateTaskId2IsLoss(
    const Plan& plan,
    std::unordered_map<int64_t, bool>* task_id2is_loss) const {
  std::unordered_map<std::string, bool> op_name2is_loss;
  GenerateOpName2IsLoss(plan, &op_name2is_loss);
  for (const TaskProto& task_proto : plan.task()) {
    bool is_loss = false;
    for (const auto& exec_sequence : task_proto.exec_sequence().exec_node()) {
      is_loss = (is_loss || op_name2is_loss[exec_sequence.op_name()]);
    }
    (*task_id2is_loss)[task_proto.id()] = is_loss;
  }
}

void PlanSGraph::GenerateOpName2IsLoss(
    const Plan& plan,
    std::unordered_map<std::string, bool>* op_name2is_loss) const {
  for (const OperatorProto& operator_proto : plan.op()) {
    const std::string& name = operator_proto.op_conf().name();
    auto op_type_case = operator_proto.op_conf().op_type_case();
    Operator* op = CreateOp(op_type_case);
    (*op_name2is_loss)[name] = op->IsLossOp();
  }
}

}  // namespace schedule
}  // namespace oneflow
