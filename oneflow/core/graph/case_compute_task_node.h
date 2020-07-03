#ifndef ONEFLOW_CORE_GRAPH_CASE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_CASE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class CaseCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CaseCompTaskNode);
  CaseCompTaskNode() = default;
  ~CaseCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kCase; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kCompute; }

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
  bool IsIndependent() const override { return true; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CASE_COMPUTE_TASK_NODE_H_
