#ifndef ONEFLOW_CORE_GRAPH_LOSS_ACCUMULATE_COMP_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_LOSS_ACCUMULATE_COMP_TASK_NODE_H_

#include "oneflow/core/graph/comp_task_node.h"

namespace oneflow {

class LossAccCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossAccCompTaskNode);
  LossAccCompTaskNode() = default;
  ~LossAccCompTaskNode() = default;

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) override;
  void InferShapeOfBlobsInProducedRegsts(TaskGraph* gph) override;
  TaskType task_type() const override { return kLossAccCompTask; }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<LossAccCompTaskNode>();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOSS_ACCUMULATE_COMP_TASK_NODE_H_
