#ifndef ONEFLOW_CORE_GRAPH_REDUCE_SCATTER_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_SCATTER_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/reduce_comp_task_node_if.h"

namespace oneflow {

class ReduceScatterCompTaskNode final : public CompTaskNode, public ReduceCompTaskNodeIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceScatterCompTaskNode);
  ReduceScatterCompTaskNode() = default;
  ~ReduceScatterCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kReduceScatter; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMix; }
  void EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) override;

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_SCATTER_COMPUTE_TASK_NODE_H_
