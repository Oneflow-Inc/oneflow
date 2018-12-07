#ifndef ONEFLOW_CORE_GRAPH_REDUCE_IDENTITY_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_IDENTITY_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/pipe_compute_task_node.h"
#include "oneflow/core/graph/reduce_comp_task_node_if.h"

namespace oneflow {

class ReduceIdentityCompTaskNode final : public CompTaskNode, public ReduceCompTaskNodeIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceIdentityCompTaskNode);
  ReduceIdentityCompTaskNode() = default;
  ~ReduceIdentityCompTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kReduceIdentity; }
  void EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) override;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_IDENTITY_TASK_NODE_H_
