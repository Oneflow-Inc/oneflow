#ifndef ONEFLOW_CORE_GRAPH_ESAC_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_ESAC_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class EsacCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EsacCompTaskNode);
  EsacCompTaskNode() = default;
  ~EsacCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kEsac; }
  CudaWorkType GetCudaWorkType() const override { return CudaWorkType::kCompute; }

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_EVERY_NTH_COMPUTE_TASK_NODE_H_
