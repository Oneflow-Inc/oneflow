#ifndef ONEFLOW_CORE_GRAPH_REDUCE_ADD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_ADD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ReduceAddCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceAddCompTaskNode);
  ReduceAddCompTaskNode() = default;
  ~ReduceAddCompTaskNode() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_ADD_COMPUTE_TASK_NODE_H_
