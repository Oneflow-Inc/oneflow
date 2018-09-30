#ifndef ONEFLOW_CORE_GRAPH_REDUCE_GATHER_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_GATHER_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/reduce_comp_task_node_if.h"

namespace oneflow {

class ReduceGatherCompTaskNode final : public CompTaskNode, public ReduceCompTaskNodeIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGatherCompTaskNode);
  ReduceGatherCompTaskNode() = default;
  ~ReduceGatherCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kReduceGather; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMix; }
  void EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) override;

 private:
  void BuildExecGphAndRegst() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_GATHER_COMPUTE_TASK_NODE_H_
