#ifndef ONEFLOW_LOGICAL_DATA_NODE_H_
#define ONEFLOW_LOGICAL_DATA_NODE_H_

#include "dag/data_node.h"

namespace oneflow {

class LogicalDataNode : public DataNode {
 public:
  DISALLOW_COPY_AND_MOVE(LogicalDataNode);
  LogicalDataNode() = default;
  ~LogicalDataNode() = default;

  void Init() {
    DataNode::Init();
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_LOGICAL_DATA_NODE_H_
