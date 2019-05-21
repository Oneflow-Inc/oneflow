#ifndef ONEFLOW_CORE_GRAPH_BOXING_COPY_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BOXING_COPY_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/register/tensor_partial_view.h"

namespace oneflow {

class BoxingCopyTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingCopyTaskNode);
  BoxingCopyTaskNode() = default;
  ~BoxingCopyTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  TaskType GetTaskType() const override { return TaskType::kBoxingCopy; }

  void BindTensorPartialViewToInDataEdge(const TaskEdge* edge, const TensorPartialView& view);
  void SetOutTensorPartialView(const TensorPartialView& out_view);
  void SetLbi(const LogicalBlobId& lbi);

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;

  HashMap<const TaskEdge*, TensorPartialView> in_data_edge2tensor_partial_view_;
  std::vector<const TaskEdge*> sorted_in_data_edge_vec_;
  TensorPartialView out_view_;
  LogicalBlobId lbi_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_COPY_TASK_NODE_H_
