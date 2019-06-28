#ifndef ONEFLOW_CORE_GRAPH_FOREIGN_IO_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_FOREIGN_IO_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ForeignIOCompTaskNode : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForeignIOCompTaskNode);
  ForeignIOCompTaskNode() = default;
  virtual ~ForeignIOCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
  bool IsMeaningLess() override { return false; }

  bool IsIndependent() const override { return true; }

 private:
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_FOREIGN_IO_COMPUTE_TASK_NODE_H_
