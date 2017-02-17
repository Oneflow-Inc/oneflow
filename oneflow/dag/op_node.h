#ifndef ONEFLOW_DAG_OP_NODE_H_
#define ONEFLOW_DAG_OP_NODE_H_

#include "dag/op_meta.h"
#include "dag/dag_node.h"

namespace oneflow {

class OpNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(OpNode);

  OpNode() = default;
  virtual ~OpNode() = default;

  void init() {
    DagNode::init();
  }

  virtual std::shared_ptr<const OpMeta> op_meta() const = 0;
 
 private:

};

} // namespace oneflow

#endif // ONEFLOW_DAG_OP_NODE_H_
