#ifndef ONEFLOW_CORE_GRAPH_SOURCE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_SOURCE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class SourceCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SourceCompTaskNode);
  SourceCompTaskNode() = default;
  ~SourceCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  TodoTaskType GetTaskType() const override { return TodoTaskType::kSource; }
  void FixThrdLocId() override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_SOURCE_COMPUTE_TASK_NODE_H_
