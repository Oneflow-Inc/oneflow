#ifndef ONEFLOW_CORE_GRAPH_BOXING_COPY_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BOXING_COPY_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/register/tensor_partial_view.h"

namespace oneflow {

enum BoxingCopyTaskMode {
  kBoxingCopyTaskModeInvalid,
  kBoxingCopyTaskModeCopy,
  kBoxingCopyTaskModeAdd,
};

class BoxingCopyTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingCopyTaskNode);
  BoxingCopyTaskNode() = default;
  ~BoxingCopyTaskNode() override = default;

  void Init(const LogicalBlobId& lbi, const TensorPartialView& out_view, BoxingCopyTaskMode mode);
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  TaskType GetTaskType() const override { return TaskType::kBoxingCopy; }
  void SetInDataEdgeView(const TaskEdge* edge, const TensorPartialView& view);

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
  OperatorConf GetBoxingOpConf();

  HashMap<const TaskEdge*, TensorPartialView> in_data_edge2view_;
  std::vector<const TaskEdge*> ordered_in_data_edges_;
  LogicalBlobId lbi_;
  TensorPartialView out_view_;
  BoxingCopyTaskMode mode_ = kBoxingCopyTaskModeInvalid;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_COPY_TASK_NODE_H_
