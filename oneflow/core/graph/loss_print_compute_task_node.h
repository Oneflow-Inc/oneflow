#ifndef ONEFLOW_CORE_GRAPH_LOSS_PRINT_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_LOSS_PRINT_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class LossPrintCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossPrintCompTaskNode);
  LossPrintCompTaskNode() = default;
  ~LossPrintCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
  TaskType GetTaskType() const override { return TaskType::kLossPrint; }
  void FixThrdId() override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOSS_PRINT_COMPUTE_TASK_NODE_H_
