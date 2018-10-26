#ifndef ONEFLOW_CORE_GRAPH_PACK_FORWARD_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_PACK_FORWARD_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/unpack_forward_task_node.h"

namespace oneflow {

class PackForwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PackForwardCompTaskNode);
  PackForwardCompTaskNode() = default;
  ~PackForwardCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kPackForward; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  void set_related_unpack(UnpackForwardCompTaskNode* val) { related_unpack_ = val; }

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;

  UnpackForwardCompTaskNode* related_unpack_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_PACK_FORWARD_TASK_NODE_H_
