#ifndef ONEFLOW_CORE_GRAPH_REDUCE_CONCAT_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_CONCAT_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/reduce_comp_task_node_if.h"

namespace oneflow {

class ReduceConcatCompTaskNode final : public CompTaskNode, public ReduceCompTaskNodeIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceConcatCompTaskNode);
  ReduceConcatCompTaskNode() = default;
  ~ReduceConcatCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kReduceConcat; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kReduceCtrl; }

  void EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) override;

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_CONCAT_COMPUTE_TASK_NODE_H_
