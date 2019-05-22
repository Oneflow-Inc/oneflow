#ifndef ONEFLOW_CORE_GRAPH_BOXING_V2_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BOXING_V2_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/register/tensor_partial_view.h"

namespace oneflow {

enum BoxingV2TaskMode {
  kBoxingV2TaskModeInvalid,
  kBoxingV2TaskModeCopy,
  kBoxingV2TaskModeAdd,
};

class BoxingV2TaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingV2TaskNode);
  BoxingV2TaskNode() = default;
  ~BoxingV2TaskNode() override = default;

  void Init(const LogicalBlobId& lbi, const TensorPartialView& out_view, BoxingV2TaskMode mode);
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  TaskType GetTaskType() const override { return TaskType::kBoxingV2; }
  void SetInDataEdgeView(const TaskEdge* edge, const TensorPartialView& view);

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
  OperatorConf GetBoxingOpConf();

  HashMap<const TaskEdge*, TensorPartialView> in_data_edge2view_;
  std::vector<const TaskEdge*> ordered_in_data_edges_;
  LogicalBlobId lbi_;
  TensorPartialView out_view_;
  BoxingV2TaskMode mode_ = kBoxingV2TaskModeInvalid;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_V2_TASK_NODE_H_
