#ifndef ONEFLOW_CORE_GRAPH_REDUCE_SCATTER_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_SCATTER_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ReduceScatterCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceScatterCompTaskNode);
  ReduceScatterCompTaskNode() = default;
  ~ReduceScatterCompTaskNode() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_SCATTER_COMPUTE_TASK_NODE_H_
