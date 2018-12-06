#ifndef ONEFLOW_CORE_GRAPH_REDUCE_INPLACE_IDENTITY_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_INPLACE_IDENTITY_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/pipe_compute_task_node.h"
#include "oneflow/core/graph/reduce_comp_task_node_if.h"

namespace oneflow {

class ReduceInplaceIdentityCompTaskNode final : public CompTaskNode, public ReduceCompTaskNodeIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceInplaceIdentityCompTaskNode);
  ReduceInplaceIdentityCompTaskNode() = default;
  ~ReduceInplaceIdentityCompTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kReduceInplaceIdentity; }
  void EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) override;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_INPLACE_IDENTITY_TASK_NODE_H_
