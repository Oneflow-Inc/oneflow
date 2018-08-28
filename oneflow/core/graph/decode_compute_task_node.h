#ifndef ONEFLOW_CORE_GRAPH_DECODE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_DECODE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class DecodeCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeCompTaskNode);
  DecodeCompTaskNode() = default;
  ~DecodeCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;

  TaskType GetTaskType() const override { return TaskType::kDecode; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_DECODE_COMPUTE_TASK_NODE_H_
