#ifndef ONEFLOW_CORE_GRAPH_ACCUMULATE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_ACCUMULATE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class AccumulateCompTaskNode : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccumulateCompTaskNode);
  AccumulateCompTaskNode() = default;
  virtual ~AccumulateCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  virtual void BuildExecGphAndRegst() override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_ACCUMULATE_COMPUTE_TASK_NODE_H_
