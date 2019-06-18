#ifndef ONEFLOW_CORE_GRAPH_MODEL_LOAD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_LOAD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class MdLoadCompTaskNode final : public CompTaskNode {
 public:
  TaskType GetTaskType() const override { return TaskType::kMdLoad; }

  bool MayBeBlocked() const override { return true; }

  void ProduceAllRegstsAndBindEdges() override;

  void ConsumeAllRegsts() override;

  void BuildExecGphAndRegst() override;

 private:
  void InferProducedDataRegstTimeShape() override {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MODEL_LOAD_COMPUTE_TASK_NODE_H_
