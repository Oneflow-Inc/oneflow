#ifndef ONEFLOW_DAG_OP_NODE_H_
#define ONEFLOW_DAG_OP_NODE_H_

#include "dag/dag_node.h"

namespace oneflow {

class OpNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(OpNode);

  virtual ~OpNode() = default;

 protected:
  OpNode() = default;
  void init() {
    DagNode::init();
  }
 
 private:

};

} // namespace oneflow

#endif // ONEFLOW_DAG_OP_NODE_H_
