#include "oneflow/core/graph/inplace_obn_graph.h"

namespace oneflow {

namespace {

void CheckSubGraph(const HashSet<const InplaceObnNode*>& nodes) {
  int32_t var_node_cnt = 0;
  int32_t is_mutable_ibn_node_cnt = 0;
  for (const auto* node : nodes) {
    if (node->op().op_conf().has_variable_conf()) {
      CHECK(node->in_edges().empty());
      CHECK_EQ(++var_node_cnt, 1);
    } else if (node->IsMutableIbn()) {
      CHECK(node->SoleInEdge()->src_node()->op().op_conf().has_variable_conf());
      CHECK_EQ(++is_mutable_ibn_node_cnt, 1);
    } else {
      // do nothing
    }
  }
}

const InplaceObnNode* GetRoot(const HashSet<const InplaceObnNode*>& nodes) {
  const InplaceObnNode* root = nullptr;
  for (const auto* node : nodes) {
    if (node->in_edges().empty()) {
      CHECK(root == nullptr);
      root = node;
    }
  }
  return root;
}

const InplaceObnNode* GetIsMutableIbnConsumer(const InplaceObnNode* node) {
  TODO();
  return nullptr;
}

void DisconnectUnReachabeAndDataMutableEdge(
    const HashSet<const InplaceObnNode*>& nodes,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    HashSet<const InplaceObnEdge*>* disabled_edges) {
  TODO();
}

void DisconnectFirstDataMutableEdgeOnEveryPath(const HashSet<const InplaceObnNode*>& nodes,
                                               HashSet<const InplaceObnEdge*>* disabled_edges) {
  TODO();
}

}  // namespace

bool NormalInplaceObnNode::IsDataMmutable() const {
  return op().OutputBlobModifier4Obn(obn()).has_mutable_inplace_ibn();
}

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
    CHECK_EQ(op->op_conf().has_variable_conf(), false);
    std::string obn = oba.bn_in_op();
    if (std::find(op->input_bns().begin(), op->input_bns().end(), oba.bn_in_op())
        != op->input_bns().end()) {
      obn += "_updated";
      CHECK(std::find(op->output_bns().begin(), op->output_bns().end(), obn)
            == op->output_bns().end());
      for (const auto& bn : op->output_bns()) {
        const auto& obn_modifier = op->OutputBlobModifier4Obn(bn);
        if (obn_modifier.has_const_inplace_ibn()) {
          CHECK(obn_modifier.const_inplace_ibn() != oba.bn_in_op());
        } else if (obn_modifier.has_mutable_inplace_ibn()) {
          CHECK(obn_modifier.mutable_inplace_ibn() != oba.bn_in_op());
        } else {
          // do nothing
        }
      }
      CHECK(op->InputBlobModifier4Ibn(oba.bn_in_op()).is_mutable());
      node = new UpdtObnInplaceObnNode(op, obn, oba.bn_in_op());
    } else if (std::find(op->output_bns().begin(), op->output_bns().end(), oba.bn_in_op())
               != op->output_bns().end()) {
      node = new NormalInplaceObnNode(op, oba.bn_in_op());
    } else {
      UNIMPLEMENTED();
    }
    AddAllocatedNode(node);
    CHECK(oba2node->emplace(oba, node).second);
  }
  CompleteObnNodes(oba2node, Op4OpName);
}

void InplaceObnGraph::CompleteObnNodes(
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
    const auto& obn_it =
        std::find_if(op->output_bns().begin(), op->output_bns().end(),
                     [&](const std::string& obn) { return op->BnInOp2Lbi(obn) == lbi; });
    CHECK(obn_it != op->output_bns().end());
    InplaceObnNode* node = nullptr;
    if (op->op_conf().has_variable_conf()) {
      new VarInplaceObnNode(op, *obn_it);
    } else {
      new NormalInplaceObnNode(op, *obn_it);
    }
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

void InplaceObnGraph::ComputeSafeInplaceObns(
    OpBlobArgList* obas,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName)
    const {
  ComputeSafeInplaceObns(IsReachableFromLbiToOpName, [&](const InplaceObnNode* node) {
    *obas->mutable_oba()->Add() = GenOpBlobArg(node->op().op_name(), node->obn());
  });
}

void InplaceObnGraph::ComputeSafeInplaceObns(
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    const std::function<void(const InplaceObnNode*)>& Handler) const {
  ForEachConnectedComponent([&](const HashSet<const InplaceObnNode*>& nodes) {
    ComputeSafeInplaceObns(nodes, IsReachableFromLbiToOpName, Handler);
  });
}

void InplaceObnGraph::ComputeSafeInplaceObns(
    const HashSet<const InplaceObnNode*>& nodes,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    const std::function<void(const InplaceObnNode*)>& Handler) const {
  CheckSubGraph(nodes);
  HashSet<const InplaceObnNode*> remainder_nodes(nodes);
  HashSet<const InplaceObnEdge*> disabled_edges;
  const InplaceObnNode* root = GetRoot(nodes);
  if (dynamic_cast<const VarInplaceObnNode*>(root) != nullptr) {
    const InplaceObnNode* updt_node = GetIsMutableIbnConsumer(root);
    if (updt_node != nullptr) {
      DisconnectUnReachabeAndDataMutableEdge(nodes, IsReachableFromLbiToOpName, &disabled_edges);
      remainder_nodes.erase(updt_node);
    } else {
      DisconnectFirstDataMutableEdgeOnEveryPath(nodes, &disabled_edges);
    }
    remainder_nodes.erase(root);
  }
  while (!remainder_nodes.empty()) {
    {
      HashSet<const InplaceObnNode*> cur_safe_inplace_obn_nodes;
      GetSafeInplaceObnNodes(remainder_nodes, disabled_edges, &cur_safe_inplace_obn_nodes);
      for (const auto* node : cur_safe_inplace_obn_nodes) { Handler(node); }
      remainder_nodes.erase(cur_safe_inplace_obn_nodes.begin(), cur_safe_inplace_obn_nodes.end());
    }
    {
      HashSet<const InplaceObnEdge*> cur_disabled_edges;
      DisconnectDataMutableEdgeByReachability(remainder_nodes, disabled_edges,
                                              IsReachableFromLbiToOpName, &cur_disabled_edges);
      disabled_edges.insert(cur_disabled_edges.begin(), cur_disabled_edges.end());
    }
    {
      HashSet<const InplaceObnEdge*> cur_disabled_edges;
      DisconnectDataMutableEdgeByReducingConficts(remainder_nodes, disabled_edges,
                                                  IsReachableFromLbiToOpName, &cur_disabled_edges);
      disabled_edges.insert(cur_disabled_edges.begin(), cur_disabled_edges.end());
    }
  }
}

void InplaceObnGraph::GetSafeInplaceObnNodes(
    const HashSet<const InplaceObnNode*>& nodes,
    const HashSet<const InplaceObnEdge*>& disabled_edges,
    HashSet<const InplaceObnNode*>* cur_disabled_nodes) const {
  TODO();
}

void InplaceObnGraph::DisconnectDataMutableEdgeByReachability(
    const HashSet<const InplaceObnNode*>& nodes,
    const HashSet<const InplaceObnEdge*>& disabled_edges,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    HashSet<const InplaceObnEdge*>* cur_disabled_edges) const {
  TODO();
}
void InplaceObnGraph::DisconnectDataMutableEdgeByReducingConficts(
    const HashSet<const InplaceObnNode*>& nodes,
    const HashSet<const InplaceObnEdge*>& disabled_edges,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    HashSet<const InplaceObnEdge*>* cur_disabled_edges) const {
  TODO();
}

}  // namespace oneflow
