#ifndef ONEFLOW_CORE_GRAPH_BOXING_IDENTITY_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BOXING_IDENTITY_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class BoxingIdentityCompTaskNode : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingIdentityCompTaskNode);
  BoxingIdentityCompTaskNode() = default;
  ~BoxingIdentityCompTaskNode() override = default;

  void Init(const CompTaskNode* src_node, const LogicalBlobId& lbi);
  TaskType GetTaskType() const override { return TaskType::kBoxingIdentity; }

 private:
  void BuildExecGphAndRegst() override;
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() final;
  void InferProducedDataRegstTimeShape() final;

  LogicalBlobId lbi_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_IDENTITY_COMPUTE_TASK_NODE_H_
