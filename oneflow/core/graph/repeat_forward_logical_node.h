#ifndef ONEFLOW_CORE_GRAPH_REPEAT_FORWARD_LOGICAL_NODE_H_
#define ONEFLOW_CORE_GRAPH_REPEAT_FORWARD_LOGICAL_NODE_H_

#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class RepeatForwardLogicalNode final : public ForwardLogicalNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RepeatForwardLogicalNode);
  RepeatForwardLogicalNode();
  ~RepeatForwardLogicalNode() override = default;

 private:
  BackwardLogicalNode* NewCorrectBackwardNode() override;
  std::string TypeName() const override { return "RepeatForward"; };
  CompTaskNode* NewCompTaskNode() const override;
  int64_t GetAreaId() const override;

  int64_t area_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REPEAT_FORWARD_LOGICAL_NODE_H_
