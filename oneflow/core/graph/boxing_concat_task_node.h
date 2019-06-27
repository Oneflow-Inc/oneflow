#ifndef ONEFLOW_CORE_GRAPH_BOXING_CONCAT_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BOXING_CONCAT_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class BoxingConcatTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingConcatTaskNode);
  BoxingConcatTaskNode() = default;
  ~BoxingConcatTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  TaskType GetTaskType() const override { return TaskType::kBoxingConcat; }
  void Init(const LogicalBlobId& lbi, int64_t machine_id, int64_t thrd_id, int64_t axis);
  void ConnectToSrc(TaskNode* src, TaskEdge* edge);

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
  OperatorConf GetConcatOpConf();

  std::vector<const TaskEdge*> ordered_in_data_edges_;
  LogicalBlobId lbi_;
  int64_t axis_ = -1;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_CONCAT_TASK_NODE_H_
