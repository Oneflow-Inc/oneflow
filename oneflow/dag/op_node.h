#ifndef ONEFLOW_DAG_OP_NODE_H_
#define ONEFLOW_DAG_OP_NODE_H_

#include "dag/op_meta.h"
#include "dag/dag_node.h"

namespace oneflow {

class OpNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(OpNode);

  OpNode() = default;
  ~OpNode() = default;

  void init(std::shared_ptr<OpMeta> op_meta) {
    DagNode::init();
    op_meta_ = op_meta;
  }

  std::shared_ptr<const OpMeta> op_meta() const { return op_meta_; }
  std::shared_ptr<OpMeta> mutable_op_meta() { return op_meta_; }
 
 private:
  std::shared_ptr<OpMeta> op_meta_;

};

} // namespace oneflow

#endif // ONEFLOW_DAG_OP_NODE_H_
