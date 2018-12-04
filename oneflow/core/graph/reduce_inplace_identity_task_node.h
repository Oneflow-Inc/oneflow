#ifndef ONEFLOW_CORE_GRAPH_REDUCE_INPLACE_IDENTITY_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_INPLACE_IDENTITY_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ReduceInplaceIdentityCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceInplaceIdentityCompTaskNode);
  ReduceInplaceIdentityCompTaskNode() = default;
  ~ReduceInplaceIdentityCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kReduceInplaceIdentity; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_INPLACE_IDENTITY_TASK_NODE_H_
