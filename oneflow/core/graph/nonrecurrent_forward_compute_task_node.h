#ifndef ONEFLOW_CORE_GRAPH_NONRECURRENT_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NONRECURRENT_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/forward_compute_task_node.h"

namespace oneflow {

class NonRecurrentForwardCompTaskNode final : public ForwardCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonRecurrentForwardCompTaskNode);
  NonRecurrentForwardCompTaskNode() = default;
  ~NonRecurrentForwardCompTaskNode() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NONRECURRENT_FORWARD_COMPUTE_TASK_NODE_H_
