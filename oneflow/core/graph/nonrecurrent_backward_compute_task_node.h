#ifndef ONEFLOW_CORE_GRAPH_NONRECURRENT_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NONRECURRENT_BACKWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/backward_compute_task_node.h"

namespace oneflow {

class NonRecurrentBackwardCompTaskNode final : public BackwardCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonRecurrentBackwardCompTaskNode);
  NonRecurrentBackwardCompTaskNode() = default;
  ~NonRecurrentBackwardCompTaskNode() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NONRECURRENT_BACKWARD_COMPUTE_TASK_NODE_H_
