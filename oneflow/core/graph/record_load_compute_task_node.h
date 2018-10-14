#ifndef ONEFLOW_CORE_GRAPH_RECORD_LOAD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_RECORD_LOAD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class RecordLoadCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordLoadCompTaskNode);
  RecordLoadCompTaskNode() = default;
  ~RecordLoadCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override {}
  void BuildExecGphAndRegst() override;
  bool IsMeaningLess() override { return false; }

  TaskType GetTaskType() const override { return TaskType::kRecordLoad; }
  bool IsPersistence() const override { return true; }

 private:
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_RECORD_LOAD_COMPUTE_TASK_NODE_H_
