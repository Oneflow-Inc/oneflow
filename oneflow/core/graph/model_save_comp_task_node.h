#ifndef ONEFLOW_CORE_GRAPH_MODEL_SAVE_COMP_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_SAVE_COMP_TASK_NODE_H_

#include "oneflow/core/graph/comp_task_node.h"

namespace oneflow {

class MdSaveCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveCompTaskNode);
  MdSaveCompTaskNode() = default;
  ~MdSaveCompTaskNode() = default;

  void ToProto(TaskProto* proto) const override {
    TaskNode::ToProto(proto);
    proto->set_parallel_policy(
        fw_task_->chain_node()->parallel_desc()->policy());
    proto->set_parallel_id(fw_task_->parallel_id());
    proto->set_parallel_num(
        fw_task_->chain_node()->parallel_desc()->parallel_num());
  }

  void set_fw_task(CompTaskNode* fw_task) { fw_task_ = fw_task; }
  CompTaskNode* fw_task() { return fw_task_; }

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) override;
  void InferShapeOfBlobsInProducedRegsts(TaskGraph* gph) override;
  bool IsMeaningLess() const override {
    return !GetConsumedRegstDesc("model");
  }

  TaskType task_type() const override { return kMdSaveCompTask; }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<MdSaveCompTaskNode>();
  }
  CompTaskNode* fw_task_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MODEL_SAVE_COMP_TASK_NODE_H_
