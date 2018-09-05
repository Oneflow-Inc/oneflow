#ifndef ONEFLOW_CORE_GRAPH_POOR_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_POOR_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class PoorCompTaskNode : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoorCompTaskNode);
  PoorCompTaskNode() = default;
  ~PoorCompTaskNode() override = default;

 private:
  void BuildExecGphAndRegst() override;
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() final;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_POOR_COMPUTE_TASK_NODE_H_
