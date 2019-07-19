#ifndef ONEFLOW_CORE_GRAPH_MODEL_INIT_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_INIT_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class MdInitCompTaskNode final : public CompTaskNode {
 public:
  TaskType GetTaskType() const override { return TaskType::kMdInit; }

  void ProduceAllRegstsAndBindEdges() override;

  void ConsumeAllRegsts() override;

  void BuildExecGphAndRegst() override;

 private:
  void InferProducedDataRegstTimeShape() override {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MODEL_INIT_COMPUTE_TASK_NODE_H_
