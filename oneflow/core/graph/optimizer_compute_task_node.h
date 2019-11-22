#ifndef ONEFLOW_CORE_GRAPH_OPTIMIZER_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_OPTIMIZER_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class OptimizerCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OptimizerCompTaskNode);
  OptimizerCompTaskNode() = default;
  ~OptimizerCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kOptimizer; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kCompute; }

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_OPTIMIZER_COMPUTE_TASK_NODE_H_
