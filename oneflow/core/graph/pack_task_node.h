#ifndef ONEFLOW_CORE_GRAPH_PACK_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_PACK_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class PackCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PackCompTaskNode);
  PackCompTaskNode() = default;
  ~PackCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kPack; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;

 private:
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_PACK_TASK_NODE_H_
