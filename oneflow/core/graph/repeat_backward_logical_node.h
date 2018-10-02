#ifndef ONEFLOW_CORE_GRAPH_REPEAT_BACKWARD_LOGICAL_NODE_H_
#define ONEFLOW_CORE_GRAPH_REPEAT_BACKWARD_LOGICAL_NODE_H_

#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class RepeatBackwardLogicalNode final : public BackwardLogicalNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RepeatBackwardLogicalNode);
  RepeatBackwardLogicalNode() = default;
  ~RepeatBackwardLogicalNode() override = default;

 private:
  std::string TypeName() const override { return "RepeatBackward"; };
  CompTaskNode* NewCompTaskNode() const override;
  int64_t GetAreaId() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REPEAT_BACKWARD_LOGICAL_NODE_H_
