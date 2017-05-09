#include "graph/exec_graph.h"

namespace oneflow {

void ExecEdge::set_lbn(const std::string& lbn) {
  lbn_ = lbn;
}

void ExecNode::ToProto(ExecNodeProto* ret) const {
  ret->set_id(node_id());
  ret->set_op_name(op_->op_name());
  for (const std::pair<std::string, RegstDesc*>& bn_regst: bn_in_op2regst_) {
    ret->mutable_bn_in_op2regst_desc_id()->insert({
        bn_regst.first, bn_regst.second->regst_desc_id()});
  }
  for (ExecEdge* edge: in_edges()) {
    ret->add_predecessor_ids(edge->src_node()->node_id());
  }
}

RegstDesc* ExecGraph::RelatedModelRegst() const {
  for (const auto& exec_node : nodes()) {
    for (const std::string& mbn : exec_node->op()->model_bns()) {
      return exec_node->GetRegstFromBnInOp(mbn);
    }
  }
  return nullptr;
}

void ExecGraph::ToProto(ExecGraphProto* ret) const {
  for (const std::unique_ptr<ExecNode>& node: nodes()) {
    node->ToProto(ret->add_exec_nodes());
  }
}

} // namespace oneflow
