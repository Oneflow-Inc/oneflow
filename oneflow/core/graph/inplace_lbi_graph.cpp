#include "oneflow/core/graph/inplace_lbi_graph.h"

namespace oneflow {

namespace {

bool IsSourceNode(const Operator& op) {
  return op.op_conf().has_variable_conf() || op.op_conf().has_constant_conf();
}

void CheckSubGraph(const HashSet<const InplaceLbiNode*>& nodes) { TODO(); }

const InplaceLbiNode* GetRoot(const HashSet<const InplaceLbiNode*>& nodes) {
  const InplaceLbiNode* root = nullptr;
  for (const auto* node : nodes) {
    if (node->in_edges().empty()) {
      CHECK(root == nullptr);
      root = node;
    }
  }
  return root;
}

const InplaceLbiNode* GetIsMutableIbnConsumer(const InplaceLbiNode* node) {
  TODO();
  return nullptr;
}

void DisconnectUnReachabeAndDataMutableEdge(
    const HashSet<const InplaceLbiNode*>& nodes,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    HashSet<const InplaceLbiEdge*>* disabled_edges) {
  TODO();
}

void DisconnectFirstDataMutableEdgeOnEveryPath(const InplaceLbiNode* root,
                                               const HashSet<const InplaceLbiNode*>& nodes,
                                               HashSet<const InplaceLbiEdge*>* disabled_edges) {
  TODO();
}

InplaceLbiNode* CreateNode(const LogicalBlobId& lbi,
                           const std::function<const Operator*(const std::string&)>& Op4OpName) {
  const Operator& op = *Op4OpName(lbi.op_name());
  if (IsSourceNode(op)) {
    return new SourceOpInplaceLbiNode(lbi);
  } else {
    if (std::find_if(op.output_bns().begin(), op.output_bns().end(),
                     [&](const std::string& obn) { return op.BnInOp2Lbi(obn) == lbi; })
        == op.output_bns().end()) {
      return new UpdateInplaceLbiNode(lbi);
    } else {
      return new NormalInplaceLbiNode(lbi);
    }
  }
}

}  // namespace

std::function<InplaceLbiNode*(const LogicalBlobId&)> InplaceLbiGraph::MakeMutFindOrCreateNode(
    std::function<const Operator*(const std::string&)> Op4OpName) {
  auto lbi2node = std::make_shared<HashMap<LogicalBlobId, InplaceLbiNode*>>();
  return [this, lbi2node, Op4OpName](const LogicalBlobId& lbi) -> InplaceLbiNode* {
    auto node_it = lbi2node->find(lbi);
    if (node_it == lbi2node->end()) {
      auto* node = CreateNode(lbi, Op4OpName);
      AddAllocatedNode(node);
      node_it = lbi2node->emplace(lbi, node).first;
    }
    return node_it->second;
  };
}

void InplaceLbiGraph::Init(const OpBlobArgList& obas,
                           const std::function<const Operator*(const std::string&)>& Op4OpName) {
  auto FindOrCreateNode = MakeMutFindOrCreateNode(Op4OpName);
  for (const auto& oba : obas.oba()) {
    const Operator& op = *Op4OpName(oba.op_name());
    LogicalBlobId lbi;
    std::string ibn;
    std::string obn;
    if (std::find(op.input_bns().begin(), op.input_bns().end(), oba.bn_in_op())
        != op.input_bns().end()) {
      ibn = oba.bn_in_op();
      obn = ibn + "_updated";
      lbi.set_op_name(op.op_name());
      lbi.set_blob_name(obn);
      CHECK(std::find_if(op.output_bns().begin(), op.output_bns().end(),
                         [&](const std::string& obn) { return op.BnInOp2Lbi(obn) == lbi; })
            == op.output_bns().end());
    } else if (std::find(op.output_bns().begin(), op.output_bns().end(), oba.bn_in_op())
               != op.output_bns().end()) {
      const auto& obn_modifier = op.OutputBlobModifier4Obn(oba.bn_in_op());
      if (obn_modifier.has_const_inplace_ibn()) {
        ibn = obn_modifier.const_inplace_ibn();
      } else if (obn_modifier.has_mutable_inplace_ibn()) {
        ibn = obn_modifier.mutable_inplace_ibn();
      } else {
        UNIMPLEMENTED();
      }
      obn = oba.bn_in_op();
      lbi = op.BnInOp2Lbi(oba.bn_in_op());
    } else {
      UNIMPLEMENTED();
    }
    auto* edge = new InplaceLbiEdge(&op, ibn, obn);
    AddAllocatedEdge(edge);
    Connect<InplaceLbiNode, InplaceLbiEdge>(FindOrCreateNode(op.BnInOp2Lbi(ibn)), edge,
                                            FindOrCreateNode(lbi));
  }
  ForEachNode([](const InplaceLbiNode* node) { CHECK_LE(node->in_edges().size(), 1); });
}

void InplaceLbiGraph::ComputeSafeInplaceObns(
    OpBlobArgList* obas,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName)
    const {
  ComputeSafeInplaceObns(IsReachableFromLbiToOpName, [&](const InplaceLbiEdge* edge) {
    CHECK_NOTNULL(dynamic_cast<const NormalInplaceLbiNode*>(edge->dst_node()));
    *obas->mutable_oba()->Add() = GenOpBlobArg(edge->op().op_name(), edge->obn());
  });
}

void InplaceLbiGraph::ComputeSafeInplaceObns(
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    const std::function<void(const InplaceLbiEdge*)>& Handler) const {
  ForEachConnectedComponent([&](const HashSet<const InplaceLbiNode*>& nodes) {
    ComputeSafeInplaceObns(nodes, IsReachableFromLbiToOpName, Handler);
  });
}

void InplaceLbiGraph::ComputeSafeInplaceObns(
    const HashSet<const InplaceLbiNode*>& nodes,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    const std::function<void(const InplaceLbiEdge*)>& Handler) const {
  CheckSubGraph(nodes);
  const InplaceLbiNode* root = GetRoot(nodes);
  HashSet<const InplaceLbiNode*> remainder_nodes(nodes);
  HashSet<const InplaceLbiEdge*> disabled_edges;
  if (dynamic_cast<const SourceOpInplaceLbiNode*>(root) != nullptr) {
    const InplaceLbiNode* updt_node = GetIsMutableIbnConsumer(root);
    if (updt_node != nullptr) {
      DisconnectUnReachabeAndDataMutableEdge(nodes, IsReachableFromLbiToOpName, &disabled_edges);
      remainder_nodes.erase(updt_node);
    }
    DisconnectFirstDataMutableEdgeOnEveryPath(root, nodes, &disabled_edges);
    remainder_nodes.erase(root);
  }
  while (!remainder_nodes.empty()) {
    {
      HashSet<const InplaceLbiEdge*> cur_disabled_edges;
      DisconnectDataMutableEdgeByReachability(remainder_nodes, disabled_edges,
                                              IsReachableFromLbiToOpName, &cur_disabled_edges);
      disabled_edges.insert(cur_disabled_edges.begin(), cur_disabled_edges.end());
    }
    {
      HashSet<const InplaceLbiEdge*> cur_disabled_edges;
      DisconnectDataMutableEdgeByReducingConficts(remainder_nodes, disabled_edges,
                                                  IsReachableFromLbiToOpName, &cur_disabled_edges);
      disabled_edges.insert(cur_disabled_edges.begin(), cur_disabled_edges.end());
    }
    {
      HashSet<const InplaceLbiEdge*> cur_safe_inplace_obn_edges;
      GetSafeInplaceObnEdges(remainder_nodes, disabled_edges, &cur_safe_inplace_obn_edges);
      for (const auto* edge : cur_safe_inplace_obn_edges) { Handler(edge); }
      disabled_edges.insert(cur_safe_inplace_obn_edges.begin(), cur_safe_inplace_obn_edges.end());
    }
    {
      HashSet<const InplaceLbiNode*> cur_disabled_nodes;
      GetDisabledNodes(remainder_nodes, disabled_edges, &cur_disabled_nodes);
      remainder_nodes.erase(cur_disabled_nodes.begin(), cur_disabled_nodes.end());
    }
  }
}

void InplaceLbiGraph::GetSafeInplaceObnEdges(
    const HashSet<const InplaceLbiNode*>& nodes,
    const HashSet<const InplaceLbiEdge*>& disabled_edges,
    HashSet<const InplaceLbiEdge*>* cur_disabled_edges) const {
  TODO();
}

void InplaceLbiGraph::GetDisabledNodes(const HashSet<const InplaceLbiNode*>& nodes,
                                       const HashSet<const InplaceLbiEdge*>& disabled_edges,
                                       HashSet<const InplaceLbiNode*>* cur_disabled_nodes) const {
  TODO();
}

void InplaceLbiGraph::DisconnectDataMutableEdgeByReachability(
    const HashSet<const InplaceLbiNode*>& nodes,
    const HashSet<const InplaceLbiEdge*>& disabled_edges,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    HashSet<const InplaceLbiEdge*>* cur_disabled_edges) const {
  TODO();
}
void InplaceLbiGraph::DisconnectDataMutableEdgeByReducingConficts(
    const HashSet<const InplaceLbiNode*>& nodes,
    const HashSet<const InplaceLbiEdge*>& disabled_edges,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    HashSet<const InplaceLbiEdge*>* cur_disabled_edges) const {
  TODO();
}

}  // namespace oneflow
