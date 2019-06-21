#ifndef ONEFLOW_CORE_GRAPH_REENTRANT_LOCK_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REENTRANT_LOCK_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ReentrantLockCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReentrantLockCompTaskNode);
  ReentrantLockCompTaskNode() = default;
  ~ReentrantLockCompTaskNode() = default;

  bool IsMeaningLess() override { return false; }
  TaskType GetTaskType() const override { return TaskType::kReentrantLock; }

 private:
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REENTRANT_LOCK_COMPUTE_TASK_NODE_H_
