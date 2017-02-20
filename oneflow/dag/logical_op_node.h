#ifndef ONEFLOW_LOGICAL_OP_NODE_H_
#define ONEFLOW_LOGICAL_OP_NODE_H_

#include "dag/op_node.h"

namespace oneflow {

class LogicalOpNode : public OpNode {
 public:
  DISALLOW_COPY_AND_MOVE(LogicalOpNode);
  LogicalOpNode() = default;
  ~LogicalOpNode() = default;

  void Init() {
    OpNode::Init();
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_LOGICAL_OP_NODE_H_
