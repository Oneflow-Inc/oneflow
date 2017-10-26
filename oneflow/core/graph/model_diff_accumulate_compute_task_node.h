#ifndef ONEFLOW_CORE_GRAPH_MODEL_DIFF_ACCUMULATE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_DIFF_ACCUMULATE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/accumulate_compute_task_node.h"

namespace oneflow {

class MdDiffAccCompTaskNode final : public AccCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdDiffAccCompTaskNode);
  MdDiffAccCompTaskNode() = default;
  ~MdDiffAccCompTaskNode() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MODEL_DIFF_ACCUMULATE_COMPUTE_TASK_NODE_H_
