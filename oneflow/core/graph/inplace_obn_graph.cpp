#include "oneflow/core/graph/inplace_obn_graph.h"

namespace oneflow {

const std::string& NormalInplaceObnNode::ibn() const {
  const auto& obn_modifier = op().OutputBlobModifier4Obn(obn());
  if (obn_modifier.has_const_inplace_ibn()) {
    return obn_modifier.const_inplace_ibn();
  } else if (obn_modifier.has_mutable_inplace_ibn()) {
    return obn_modifier.mutable_inplace_ibn();
  } else {
    UNIMPLEMENTED();
  }
}

void InplaceObnGraph::InitNodes(
    HashMap<OpBlobArg, InplaceObnNode*>* oba2node, const OpBlobArgList& obas,
    const std::function<const Operator*(const std::string&)>& Op4OpName) {
  for (const auto& oba : obas.oba()) {
    InplaceObnNode* node = nullptr;
    const Operator* op = Op4OpName(oba.op_name());
    std::string obn = oba.bn_in_op();
    if (std::find(op->input_bns().begin(), op->input_bns().end(), oba.bn_in_op())
        != op->input_bns().end()) {
      obn += "_updated";
      CHECK(std::find(op->output_bns().begin(), op->output_bns().end(), obn)
            == op->output_bns().end());
      node = new FakeObnInplaceObnNode(op, obn, oba.bn_in_op());
    } else if (std::find(op->output_bns().begin(), op->output_bns().end(), oba.bn_in_op())
               != op->output_bns().end()) {
      node = new NormalInplaceObnNode(op, oba.bn_in_op());
    } else {
      UNIMPLEMENTED();
    }
    AddAllocatedNode(node);
    CHECK(oba2node->emplace(oba, node).second);
  }
  CompleteVariableObnNodes(oba2node, Op4OpName);
}

void InplaceObnGraph::CompleteVariableObnNodes(
    HashMap<OpBlobArg, InplaceObnNode*>* oba2node,
    const std::function<const Operator*(const std::string&)>& Op4OpName) {
  HashSet<LogicalBlobId> existed_lbis;
  ForEachNode([&](const InplaceObnNode* node) {
    if (dynamic_cast<const NormalInplaceObnNode*>(node) != nullptr) {
      CHECK(existed_lbis.emplace(node->op().BnInOp2Lbi(node->obn())).second);
    }
  });
  HashSet<LogicalBlobId> todo_lbis;
  ForEachNode([&](const InplaceObnNode* node) {
    const auto& lbi = node->op().BnInOp2Lbi(node->ibn());
    if (existed_lbis.find(lbi) == existed_lbis.end()) { todo_lbis.emplace(lbi); }
  });
  for (const auto& lbi : todo_lbis) {
    const Operator* op = Op4OpName(lbi.op_name());
    CHECK(op->op_conf().has_variable_conf());
    const auto& obn_it =
        std::find_if(op->output_bns().begin(), op->output_bns().end(),
                     [&](const std::string& obn) { return op->BnInOp2Lbi(obn) == lbi; });
    CHECK(obn_it != op->output_bns().end());
    InplaceObnNode* node = new NormalInplaceObnNode(op, *obn_it);
    AddAllocatedNode(node);
    CHECK(oba2node->emplace(GenOpBlobArg(op->op_name(), *obn_it), node).second);
  }
}

void InplaceObnGraph::InitEdges(
    const HashMap<OpBlobArg, InplaceObnNode*>& oba2node, const OpBlobArgList& obas,
    const std::function<const Operator*(const std::string&)>& Op4OpName) {
  HashMap<LogicalBlobId, InplaceObnNode*> lbi2normal_node;
  for (const auto& pair : oba2node) {
    if (dynamic_cast<NormalInplaceObnNode*>(pair.second) != nullptr) {
      const auto& lbi = Op4OpName(pair.first.op_name())->BnInOp2Lbi(pair.first.bn_in_op());
      CHECK(lbi2normal_node.emplace(lbi, pair.second).second);
    }
  }
  for (const auto& pair : oba2node) {
    const auto& in_lbi = Op4OpName(pair.first.op_name())->BnInOp2Lbi(pair.second->ibn());
    auto* edge = new InplaceObnEdge();
    AddAllocatedEdge(edge);
    Connect<InplaceObnNode, InplaceObnEdge>(lbi2normal_node.at(in_lbi), edge, pair.second);
  }
}

void InplaceObnGraph::Init(const OpBlobArgList& obas,
                           const std::function<const Operator*(const std::string&)>& Op4OpName) {
  HashMap<OpBlobArg, InplaceObnNode*> oba2node;
  InitNodes(&oba2node, obas, Op4OpName);
  InitEdges(oba2node, obas, Op4OpName);
  ForEachNode([](const InplaceObnNode* node) { CHECK_LE(node->in_edges().size(), 1); });
}

void InplaceObnGraph::ComputeSafeInplaceObns(OpBlobArgList* obas) const { TODO(); }

}  // namespace oneflow
