#include "graph/exec_graph.h"

namespace oneflow {

void ExecEdge::set_lbn(const std::string& lbn) {
  lbn_ = lbn;
}
ExecNodeProto ExecNode::ToProto() const {
  ExecNodeProto exnode;
  exnode.set_id(node_id());
  exnode.set_op_name(op_->op_name());
  for (const std::pair<std::string, RegstDesc*>& bn_regst: bn_in_op2regst_) {
    exnode.mutable_bn_in_op2regst_desc_id()->insert({
        bn_regst.first, bn_regst.second->regst_desc_id()});
  }
  for (ExecEdge* edge: in_edges()) {
    exnode.add_predecessor_ids(edge->src_node()->node_id());
  }
  return exnode;
}

ExecGraphProto ExecGraph::ToProto() const {
  ExecGraphProto exgraph;
  for (const std::unique_ptr<ExecNode>& node: nodes()) {
    *(exgraph.add_exec_nodes()) = node->ToProto();
  }
  return exgraph;
}

} // namespace oneflow
