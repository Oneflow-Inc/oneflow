#ifndef ONEFLOW_CORE_GRAPH_ACCUMULATE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_ACCUMULATE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class AccCompTaskNode : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccCompTaskNode);
  AccCompTaskNode() = default;
  virtual ~AccCompTaskNode() = default;

  void NewAllProducedRegst() override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_ACCUMULATE_COMPUTE_TASK_NODE_H_
