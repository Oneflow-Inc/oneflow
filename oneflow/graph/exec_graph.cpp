#include "graph/exec_graph.h"

namespace oneflow {

void ExecEdge::set_lbn(const std::string& lbn) {
  lbn_ = lbn;
}
ExecNodeProto ExecNode::ToProto() const {
  ExecNodeProto exnode;
  exnode.set_id(node_id());
  exnode.set_op_name(op_->op_name());
  HashMap<std::string, int64_t> bn2id;
  for (auto bn_regst: bn_in_op2regst_) {
    bn2id.emplace(bn_regst.first, 
                                   bn_regst.second->regst_desc_id());
  }
  using RetType = google::protobuf::Map<std::string, google::protobuf::uint64>;
  *(exnode.mutable_bn_in_op2regst_desc_id()) = RetType(bn2id.begin(),
                                                       bn2id.end());
  for (auto edge: in_edges()) {
    exnode.add_predecessor_ids(edge->src_node()->node_id());
  }
  return exnode;
}

ExecGraphProto ExecGraph::ToProto() const {
  ExecGraphProto exgraph;
  for (std::size_t i = 0; i < nodes().size(); ++i) {
    *(exgraph.add_exec_nodes()) = nodes().at(i)->ToProto();
  }
  return exgraph;
}

} // namespace oneflow
