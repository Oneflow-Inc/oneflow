#ifndef ONEFLOW_CORE_GRAPH_LOSS_RECORD_COMP_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_LOSS_RECORD_COMP_TASK_NODE_H_

#include "oneflow/core/graph/comp_task_node.h"

namespace oneflow {

class LossRecordCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossRecordCompTaskNode);
  LossRecordCompTaskNode() = default;
  ~LossRecordCompTaskNode() = default;

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) override;
  void InferBlobDescInProducedRegsts(TaskGraph* gph) override;
  bool IsMeaningLess() const override {
    return !GetConsumedRegstDesc("loss_acc");
  }
  TaskType task_type() const override { return kLossRecordCompTask; }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<LossRecordCompTaskNode>();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOSS_RECORD_COMP_TASK_NODE_H_
