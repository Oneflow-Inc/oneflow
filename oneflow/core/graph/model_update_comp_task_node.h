#ifndef ONEFLOW_CORE_GRAPH_MODEL_UPDATE_COMP_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_UPDATE_COMP_TASK_NODE_H_

#include "oneflow/core/graph/data_comp_task_node.h"

namespace oneflow {

class MdUpdtCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtCompTaskNode);
  MdUpdtCompTaskNode() = default;
  ~MdUpdtCompTaskNode() = default;

  void ToProto(TaskProto* proto) const override {
    TaskNode::ToProto(proto);
    proto->set_parallel_policy(
        fw_task_->chain_node()->parallel_desc()->policy());
    proto->set_parallel_id(fw_task_->parallel_id());
    proto->set_parallel_num(
        fw_task_->chain_node()->parallel_desc()->parallel_num());
    int64_t related_save_task_id = -1;
    for (const auto& pair : produced_regst_descs()) {
      for (const TaskNode* subscriber : pair.second->subscribers()) {
        if (dynamic_cast<const DataCompTaskNode*>(subscriber) == nullptr) {
          CHECK_EQ(related_save_task_id, -1);
          related_save_task_id = subscriber->task_id();
        }
      }
    }
    proto->set_related_save_task_id(related_save_task_id);
  }

  void set_fw_task(CompTaskNode* fw_task) { fw_task_ = fw_task; }
  CompTaskNode* fw_task() { return fw_task_; }

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) override;
  void InferShapeOfBlobsInProducedRegsts(TaskGraph* gph) override;
  TaskType task_type() const override { return kMdUpdtCompTask; }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<MdUpdtCompTaskNode>();
  }
  CompTaskNode* fw_task_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MODEL_UPDATE_COMP_TASK_NODE_H_
