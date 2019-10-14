#ifndef ONEFLOW_CORE_GRAPH_REDUCE_SPLIT_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_SPLIT_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/reduce_comp_task_node_if.h"

namespace oneflow {

class ReduceSplitCompTaskNode final : public CompTaskNode, public ReduceCompTaskNodeIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceSplitCompTaskNode);
  ReduceSplitCompTaskNode() = default;
  ~ReduceSplitCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kReduceSplit; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kReduceCtrl; }
  void EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) override;

  TaskNode* GetPrevReduceTaskNode(TaskType task_type);

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_SPLIT_COMPUTE_TASK_NODE_H_
