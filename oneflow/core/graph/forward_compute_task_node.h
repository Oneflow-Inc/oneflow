#ifndef ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class FwCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FwCompTaskNode);
  FwCompTaskNode() = default;
  ~FwCompTaskNode() = default;

  void NewAllProducedRegst() override;
  TodoTaskType GetTaskType() const override { return TodoTaskType::kForward; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_
