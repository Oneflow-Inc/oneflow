#include "oneflow/core/schedule/plan_sgraph.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/operator/operator_manager.h"
namespace oneflow {
namespace schedule {

void PlanSGraph::InitRegstDesc(const Plan& plan) {
  std::unordered_map<int64_t, const TaskProto*> id2task_proto;
  for (const TaskProto& task_proto : plan.task()) {
    id2task_proto[task_proto.id()] = &task_proto;
  }
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      uint64_t producer_id = pair.second.producer_task_id();
      STask* producer = node_mgr<STask>().Find(producer_id);
      CHECK(producer);
      uint64_t regst_desc_id = pair.second.regst_desc_id();
      SRegstDesc* regst_desc =
          mut_node_mgr<SRegstDesc>()->CreateIfNotFound(regst_desc_id);
      regst_desc->mut_origin_regst_count() = pair.second.register_num();
      if (task_proto.type() == kMdUpdtCompTask) {
        regst_desc->mut_min_regst_count() = 3u;
      }
      mut_produced_regst_desc_mgr()->CreateIfNotFound(producer, regst_desc);
      if (!pair.second.consumer_task_id().size()) {
        mut_subscribed_regst_desc_mgr()->CreateIfNotFound(producer, regst_desc);
      }
      for (int64_t consumer_id : pair.second.consumer_task_id()) {
        STask* consumer = node_mgr<STask>().Find(consumer_id);
        CHECK(consumer);
        const TaskProto* consumer_task_proto = id2task_proto[consumer_id];
        CHECK(consumer);
        CHECK(producer);
        CHECK(consumer_task_proto);
        if (!(task_proto.type() == kMdUpdtCompTask
              && consumer_task_proto->type() == kDataCompTask)) {
          mut_subscribed_regst_desc_mgr()->CreateIfNotFound(consumer,
                                                            regst_desc);
          mut_arc_mgr()->CreateIfNotFound(producer, consumer);
        }
      }
    }
  }
}

void PlanSGraph::InitTask(const Plan& plan) {
  for (const TaskProto& task_proto : plan.task()) {
    uint64_t task_id = task_proto.id();
    float workload = task_proto.exec_sequence().exec_node().size();
    std::string name = TaskType_Name(task_proto.type()) + "\\n"
                       + std::to_string(task_id) + "\\n"
                       + +(task_proto.is_forward() ? "fw" : "bp");
    //	STask
    STask* node = mut_node_mgr<STask>()->CreateWithId(task_id, name, workload);
    CHECK(node);
    mut_children_arc_mgr()->CreateIfNotFound(this, node);
    bool is_copy_hd = (task_proto.type() == kCopyHdTask);
    //	SDevice
    uint64_t device_id =
        IDMgr::Singleton()->UUDeviceId4TaskId(task_proto.id(), is_copy_hd);
    std::string device_name = std::to_string(device_id);
    float defval = 0.0;
    std::string dev_name = std::to_string(device_id);
    SDevice* device =
        mut_node_mgr<SDevice>()->CreateIfNotFound(device_id, dev_name, defval);
    mut_device_arc_mgr()->CreateIfNotFound(node, device);
    device->mut_bandwidth() += workload;
  }
}

void PlanSGraph::InitLoss(const Plan& plan) {
  std::unordered_map<int64_t, bool> task_id2is_loss;
  GenerateTaskId2IsLoss(plan, &task_id2is_loss);
  for (const TaskProto& task_proto : plan.task()) {
    if (task_id2is_loss[task_proto.id()]) {
      STask* loss = node_mgr<STask>().Find(task_proto.id());
      CHECK(loss);
      mut_loss_arc_mgr()->CreateIfNotFound(this, loss);
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
