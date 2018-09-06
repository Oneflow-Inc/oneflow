#ifndef ONEFLOW_CORE_GRAPH_REDUCE_GLOBAL_ADD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_GLOBAL_ADD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/reduce_comp_task_node_if.h"

namespace oneflow {

class ReduceGlobalAddCompTaskNode final : public CompTaskNode, public ReduceCompTaskNodeIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGlobalAddCompTaskNode);
  ReduceGlobalAddCompTaskNode() = default;
  ~ReduceGlobalAddCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kReduceGlobalAdd; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kMix; }
  void EnableMemSharingInReduce(
      std::function<void(RegstDesc* regst, int64_t offset)> EnableMemSharing4Regst) override;

 private:
  void BuildExecGphAndRegst() override;

  PbRf<int64_t> in_parallel_ids_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_GLOBAL_ADD_COMPUTE_TASK_NODE_H_
