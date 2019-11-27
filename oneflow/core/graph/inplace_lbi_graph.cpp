#include "oneflow/core/graph/inplace_lbi_graph.h"

namespace oneflow {

namespace {

bool IsSourceNode(const Operator& op) {
  return op.op_conf().has_variable_conf() || op.op_conf().has_constant_conf();
}

void CheckSubGraph(const HashSet<const InplaceLbiNode*>& nodes) {
  size_t source_op_node_cnt = 0;
  size_t updt_node_cnt = 0;
  size_t source_cnt = 0;
  for (const auto* node : nodes) {
    if (node->in_edges().empty()) { CHECK_EQ(++source_cnt, 1); }
    if (dynamic_cast<const SourceOpInplaceLbiNode*>(node) != nullptr) {
      CHECK_EQ(++source_op_node_cnt, 1);
      CHECK(node->in_edges().empty());
    }
    if (dynamic_cast<const UpdateInplaceLbiNode*>(node) != nullptr) {
      CHECK_EQ(++updt_node_cnt, 1);
      CHECK_NOTNULL(dynamic_cast<const SourceOpInplaceLbiNode*>(node->SoleInEdge()->src_node()));
    }
  }
}

const InplaceLbiNode* GetRoot(const HashSet<const InplaceLbiNode*>& nodes,
                              const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge) {
  const InplaceLbiNode* root = nullptr;
  for (const InplaceLbiNode* node : nodes) {
    if (node->GetValidInEdge(IsValidEdge) == nullptr) {
      CHECK_ISNULL(root);
      root = node;
    }
  }
  return root;
}

const InplaceLbiNode* FindSoleIsMutableIbnConsumer(const SourceOpInplaceLbiNode* node) {
  const InplaceLbiNode* ret = nullptr;
  for (const InplaceLbiEdge* edge : node->out_edges()) {
    if (dynamic_cast<const UpdateInplaceLbiNode*>(edge->dst_node()) != nullptr) {
      CHECK_ISNULL(ret);
      ret = edge->dst_node();
    }
  }
  return ret;
}

InplaceLbiNode* CreateNode(const LogicalBlobId& lbi,
                           const std::function<const Operator*(const std::string&)>& Op4OpName) {
  const Operator& op = *Op4OpName(lbi.op_name());
  if (IsSourceNode(op)) {
    return new SourceOpInplaceLbiNode(lbi);
  } else if (std::find_if(op.output_bns().begin(), op.output_bns().end(),
                          [&](const std::string& obn) { return op.BnInOp2Lbi(obn) == lbi; })
             != op.output_bns().end()) {
    return new NormalInplaceLbiNode(lbi);
  } else {
    return new UpdateInplaceLbiNode(lbi);
  }
}

void GetUnconnectedNodes(const HashSet<const InplaceLbiNode*>& nodes,
                         const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge,
                         HashSet<const InplaceLbiNode*>* cur_disabled_nodes) {
  for (const InplaceLbiNode* node : nodes) {
    size_t cnt = 0;
    for (const InplaceLbiEdge* edge : node->in_edges()) { cnt += IsValidEdge(edge); }
    for (const InplaceLbiEdge* edge : node->out_edges()) { cnt += IsValidEdge(edge); }
    if (cnt == 0) { CHECK(cur_disabled_nodes->emplace(node).second); }
  }
}

const InplaceLbiNode* GetFirstDiffNode(const std::vector<const InplaceLbiNode*>& lhs,
                                       const std::vector<const InplaceLbiNode*>& rhs) {
  FOR_RANGE(int32_t, i, 0, std::min(lhs.size(), rhs.size())) {
    if (lhs.at(i) != rhs.at(i)) { return lhs.at(i); }
  }
  return nullptr;
};

std::function<void(const InplaceLbiNode*, const std::function<void(const InplaceLbiNode*)>&)>
GetForEachValidInNode(const HashSet<const InplaceLbiNode*>* nodes,
                      std::function<bool(const InplaceLbiEdge*)> IsValidEdge) {
  return [nodes, IsValidEdge](const InplaceLbiNode* node,
                              const std::function<void(const InplaceLbiNode*)>& Handler) {
    const InplaceLbiEdge* in_edge = node->GetValidInEdge(IsValidEdge);
    if (in_edge == nullptr) { return; }
    if (nodes->find(in_edge->src_node()) != nodes->end()) { Handler(in_edge->src_node()); }
  };
}

std::function<void(const InplaceLbiNode*, const std::function<void(const InplaceLbiNode*)>&)>
GetForEachValidOutNode(const HashSet<const InplaceLbiNode*>* nodes,
                       std::function<bool(const InplaceLbiEdge*)> IsValidEdge) {
  return [nodes, IsValidEdge](const InplaceLbiNode* node,
                              const std::function<void(const InplaceLbiNode*)>& Handler) {
    node->ForEachNodeOnValidOutEdge(IsValidEdge, [&](const InplaceLbiNode* out_node) {
      if (nodes->find(out_node) != nodes->end()) { Handler(out_node); }
    });
  };
}

bool IsOtherIbnBoundToOneOfLbis(const HashSet<LogicalBlobId>& lbis, const InplaceLbiEdge* edge) {
  const Operator& op = edge->op();
  for (const std::string& ibn : op.input_bns()) {
    if (ibn != edge->ibn() && lbis.find(op.BnInOp2Lbi(ibn)) != lbis.end()) { return true; }
  }
  return false;
}

void RemoveUnconnectedNodes(HashSet<const InplaceLbiNode*>* nodes,
                            const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge) {
  HashSet<const InplaceLbiNode*> cur_disabled_nodes;
  GetUnconnectedNodes(*nodes, IsValidEdge, &cur_disabled_nodes);
  for (const auto* node : cur_disabled_nodes) { nodes->erase(node); }
}

}  // namespace

const InplaceLbiEdge* InplaceLbiNode::GetValidInEdge(
    const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge) const {
  if (!in_edges().empty() && IsValidEdge(SoleInEdge())) { return SoleInEdge(); }
  return nullptr;
}

const InplaceLbiEdge* InplaceLbiNode::GetSoleValidInEdge(
    const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge) const {
  const auto* edge = GetValidInEdge(IsValidEdge);
  CHECK_NOTNULL(edge);
  return edge;
}

void InplaceLbiNode::ForEachNodeOnValidOutEdge(
    const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge,
    const std::function<void(const InplaceLbiNode*)>& Handler) const {
  for (const auto* edge : out_edges()) {
    if (IsValidEdge(edge)) { Handler(edge->dst_node()); }
  }
}

bool InplaceLbiNode::IsMutRef(const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge) const {
  UNIMPLEMENTED();
}

bool InplaceLbiNode::IsConstRef(
    const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge) const {
  return !IsMutRef(IsValidEdge);
}

bool NormalInplaceLbiNode::IsMutRef(
    const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge) const {
  const InplaceLbiEdge* in_edge = GetValidInEdge(IsValidEdge);
  return in_edge != nullptr && in_edge->IsMutRef();
}

bool InplaceLbiEdge::IsMutRef() const {
  CHECK_NOTNULL(dynamic_cast<const NormalInplaceLbiNode*>(dst_node()));
  const OutputBlobModifier& obn_modifier = op().OutputBlobModifier4Obn(obn());
  CHECK(obn_modifier.has_mutable_inplace_ibn() || obn_modifier.has_const_inplace_ibn());
  return obn_modifier.has_mutable_inplace_ibn();
}

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
  CHECK(!FindFirstNontrivialSCC());
}

void InplaceLbiGraph::ComputeSafeInplaceObns(
    OpBlobArgList* obas,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName)
    const {
  ComputeSafeInplaceEdges(IsReachableFromLbiToOpName, [&](const InplaceLbiEdge* edge) {
    CHECK_NOTNULL(dynamic_cast<const NormalInplaceLbiNode*>(edge->dst_node()));
    *obas->mutable_oba()->Add() = GenOpBlobArg(edge->op().op_name(), edge->obn());
  });
}

void InplaceLbiGraph::ComputeSafeInplaceEdges(
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    const std::function<void(const InplaceLbiEdge*)>& Handler) const {
  ForEachConnectedComponent([&](const HashSet<const InplaceLbiNode*>& nodes) {
    ComputeSafeInplaceEdges(nodes, IsReachableFromLbiToOpName, Handler);
  });
}

void InplaceLbiGraph::ForEachSafeInplaceEdgeInSourceOpSubTree(
    const HashSet<const InplaceLbiNode*>& nodes,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    const std::function<void(const InplaceLbiEdge*)>& Handler,
    HashSet<const InplaceLbiEdge*>* disabled_edges) const {
  disabled_edges->clear();
  auto IsValidEdge = [&](const InplaceLbiEdge* edge) {
    return disabled_edges->find(edge) == disabled_edges->end();
  };
  const InplaceLbiNode* root = GetRoot(nodes, [](const InplaceLbiEdge*) { return true; });
  const auto* source_op_root = dynamic_cast<const SourceOpInplaceLbiNode*>(root);
  if (source_op_root != nullptr) {
    const InplaceLbiNode* updt_node = FindSoleIsMutableIbnConsumer(source_op_root);
    if (updt_node != nullptr) {
      HashSet<const InplaceLbiEdge*> cur_disabled_edges;
      FixConstRefOrMutRefConflictsToUpdtNode(nodes, IsReachableFromLbiToOpName,
                                             &cur_disabled_edges);
      disabled_edges->insert(cur_disabled_edges.begin(), cur_disabled_edges.end());
    }
    {
      HashSet<const InplaceLbiEdge*> cur_disabled_edges;
      FixMutRefConflictsFromSourceOpNode(source_op_root, IsValidEdge, &cur_disabled_edges);
      disabled_edges->insert(cur_disabled_edges.begin(), cur_disabled_edges.end());
    }
    {
      // disconnect edges in the subtree containning `root`
      HashSet<const InplaceLbiEdge*> cur_disabled_edges;
      auto ForEachNext = GetForEachValidOutNode(&nodes, IsValidEdge);
      BfsForEachNode({root}, ForEachNext, [&](const InplaceLbiNode* node) {
        const InplaceLbiEdge* in_edge = node->GetValidInEdge(IsValidEdge);
        if (in_edge != nullptr) { CHECK(cur_disabled_edges.emplace(in_edge).second); }
        if (dynamic_cast<const NormalInplaceLbiNode*>(node) != nullptr) {
          CHECK_NOTNULL(in_edge);
          CHECK(node->IsConstRef(IsValidEdge));
          Handler(in_edge);
        }
      });
      disabled_edges->insert(cur_disabled_edges.begin(), cur_disabled_edges.end());
    }
  }
}

void InplaceLbiGraph::ComputeSafeInplaceEdges(
    const HashSet<const InplaceLbiNode*>& nodes,
    const std::function<bool(const LogicalBlobId&, const std::string&)>& IsReachableFromLbiToOpName,
    const std::function<void(const InplaceLbiEdge*)>& Handler) const {
  CheckSubGraph(nodes);
  HashSet<const InplaceLbiNode*> remainder_nodes(nodes);
  HashSet<const InplaceLbiEdge*> disabled_edges;
  {
    // compute safe inplace edges in the subtree containning SourceOpInplaceLbiNode as root
    HashSet<const InplaceLbiEdge*> cur_disabled_edges;
    ForEachSafeInplaceEdgeInSourceOpSubTree(remainder_nodes, IsReachableFromLbiToOpName, Handler,
                                            &cur_disabled_edges);
    disabled_edges.insert(cur_disabled_edges.begin(), cur_disabled_edges.end());
  }
  auto IsValidEdge = [&](const InplaceLbiEdge* edge) {
    return remainder_nodes.find(edge->src_node()) != remainder_nodes.end()
           && remainder_nodes.find(edge->dst_node()) != remainder_nodes.end()
           && disabled_edges.find(edge) == disabled_edges.end();
  };
  RemoveUnconnectedNodes(&remainder_nodes, IsValidEdge);
  size_t dead_loop_check = remainder_nodes.size();
  while (!remainder_nodes.empty()) {
    ForEachTree(remainder_nodes, IsValidEdge, [&](const HashSet<const InplaceLbiNode*>& nodes) {
      const InplaceLbiEdge* cur_disabled_edge =
          FindFirstInterOpRefConflictMutRefEdge(nodes, IsValidEdge, IsReachableFromLbiToOpName);
      if (cur_disabled_edge != nullptr) { disabled_edges.insert(cur_disabled_edge); }
    });
    ForEachTree(remainder_nodes, IsValidEdge, [&](const HashSet<const InplaceLbiNode*>& nodes) {
      const InplaceLbiEdge* cur_disabled_edge =
          FindFirstConstRefConflictMutRefEdge(nodes, IsValidEdge, IsReachableFromLbiToOpName);
      if (cur_disabled_edge != nullptr) { disabled_edges.insert(cur_disabled_edge); }
    });
    ForEachTree(remainder_nodes, IsValidEdge, [&](const HashSet<const InplaceLbiNode*>& nodes) {
      const InplaceLbiEdge* cur_disabled_edge =
          FindFirstIntraOpRefConflictMutRefEdge(nodes, IsValidEdge);
      if (cur_disabled_edge != nullptr) { disabled_edges.insert(cur_disabled_edge); }
    });
    {
      HashSet<const InplaceLbiEdge*> cur_safe_inplace_obn_edges;
      GetSafeInplaceObnEdges(remainder_nodes, IsValidEdge, IsReachableFromLbiToOpName,
                             &cur_safe_inplace_obn_edges);
      for (const auto* edge : cur_safe_inplace_obn_edges) { Handler(edge); }
      disabled_edges.insert(cur_safe_inplace_obn_edges.begin(), cur_safe_inplace_obn_edges.end());
    }
    RemoveUnconnectedNodes(&remainder_nodes, IsValidEdge);
    CHECK_GE(--dead_loop_check, 0);
  }
}

void InplaceLbiGraph::FindAllEdges(const HashSet<const InplaceLbiNode*>& nodes,
                                   const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge,
                                   HashSet<const InplaceLbiEdge*>* cur_disabled_edges) const {
  for (const auto* node : nodes) {
    node->ForEachNodeOnValidOutEdge(IsValidEdge, [&](const InplaceLbiNode* out_node) {
      CHECK(cur_disabled_edges->emplace(out_node->GetSoleValidInEdge(IsValidEdge)).second);
    });
  }
}

const InplaceLbiEdge* InplaceLbiGraph::FindFirstIntraOpRefConflictMutRefEdge(
    const HashSet<const InplaceLbiNode*>& nodes,
    const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge) const {
  const InplaceLbiEdge* ret = nullptr;
  HashSet<LogicalBlobId> lbis;
  for (const auto* node : nodes) { CHECK(lbis.insert(node->lbi()).second); }

  const auto* root = GetRoot(nodes, IsValidEdge);
  auto ForEachInNode = GetForEachValidInNode(&nodes, IsValidEdge);
  auto ForEachOutNode = GetForEachValidOutNode(&nodes, IsValidEdge);
  TopoForEachNode({root}, ForEachInNode, ForEachOutNode, [&](const InplaceLbiNode* node) {
    if (ret != nullptr) { return; }
    if (node->IsMutRef(IsValidEdge) && IsOtherIbnBoundToOneOfLbis(lbis, node->SoleInEdge())) {
      ret = node->SoleInEdge();
    }
  });
  return ret;
}

bool InplaceLbiGraph::IsConstRefConflictMutRefNode(
    const InplaceLbiNode* mut_ref_node, const HashSet<const InplaceLbiNode*>& nodes,
    const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge,
    const std::function<bool(const LogicalBlobId&, const std::string&)>&
        IsLbiAllConsumerReachableToOpName) const {
  CHECK(mut_ref_node->IsMutRef(IsValidEdge));
  auto ForEachNext = [&](const InplaceLbiNode* node,
                         const std::function<void(const InplaceLbiNode*)>& Handler) {
    node->ForEachNodeOnValidOutEdge(IsValidEdge, [&](const InplaceLbiNode* out_node) {
      if (out_node != mut_ref_node) { Handler(out_node); }
    });
  };
  bool conflict = false;
  const auto& op_name = mut_ref_node->lbi().op_name();
  BfsForEachNode({GetRoot(nodes, IsValidEdge)}, ForEachNext, [&](const InplaceLbiNode* node) {
    conflict = conflict || !IsLbiAllConsumerReachableToOpName(node->lbi(), op_name);
  });
  return conflict;
}

const InplaceLbiEdge* InplaceLbiGraph::FindFirstConstRefConflictMutRefEdge(
    const HashSet<const InplaceLbiNode*>& nodes,
    const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge,
    const std::function<bool(const LogicalBlobId&, const std::string&)>&
        IsLbiAllConsumerReachableToOpName) const {
  const InplaceLbiNode* root = GetRoot(nodes, IsValidEdge);
  auto ForEachInNode = GetForEachValidInNode(&nodes, IsValidEdge);
  auto ForEachOutNode = GetForEachValidOutNode(&nodes, IsValidEdge);
  const InplaceLbiEdge* ret = nullptr;
  TopoForEachNode({root}, ForEachInNode, ForEachOutNode, [&](const InplaceLbiNode* node) {
    if (ret != nullptr) { return; }
    if (node->IsMutRef(IsValidEdge)
        && IsConstRefConflictMutRefNode(node, nodes, IsValidEdge,
                                        IsLbiAllConsumerReachableToOpName)) {
      ret = node->GetValidInEdge(IsValidEdge);
    }
  });
  return ret;
}

const InplaceLbiEdge* InplaceLbiGraph::FindFirstInterOpRefConflictMutRefEdge(
    const HashSet<const InplaceLbiNode*>& nodes,
    const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge,
    const std::function<bool(const LogicalBlobId&, const std::string&)>&
        IsLbiAllConsumerReachableToOpName) const {
  HashSet<const InplaceLbiNode*> mut_ref_nodes;
  HashMap<const InplaceLbiNode*, std::vector<const InplaceLbiNode*>> node2mut_ref_ancestors;
  {
    const InplaceLbiNode* root = GetRoot(nodes, IsValidEdge);
    auto ForEachInNode = GetForEachValidInNode(&nodes, IsValidEdge);
    auto ForEachOutNode = GetForEachValidOutNode(&nodes, IsValidEdge);
    TopoForEachNode({root}, ForEachInNode, ForEachOutNode, [&](const InplaceLbiNode* node) {
      if (node->IsMutRef(IsValidEdge)) { mut_ref_nodes.insert(node); }
      size_t in_edges_size_check = 0;
      ForEachInNode(node, [&](const InplaceLbiNode* in_node) {
        node2mut_ref_ancestors[node] = node2mut_ref_ancestors[in_node];
        if (in_node->IsMutRef(IsValidEdge)) { node2mut_ref_ancestors[node].push_back(in_node); }
        CHECK_EQ(++in_edges_size_check, 1);
      });
    });
  }
  std::vector<const InplaceLbiNode*> last_mut_ref_nodes;
  {
    HashMap<const InplaceLbiNode*, size_t> mut_ref_node2descendents_size;
    for (const InplaceLbiNode* descendent : mut_ref_nodes) {
      for (const InplaceLbiNode* ancestor : node2mut_ref_ancestors.at(descendent)) {
        ++mut_ref_node2descendents_size[ancestor];
      }
    }
    for (const InplaceLbiNode* node : mut_ref_nodes) {
      if (mut_ref_node2descendents_size[node] == 0) { last_mut_ref_nodes.push_back(node); }
    }
  }
  if (last_mut_ref_nodes.size() <= 1) { return nullptr; }
  const InplaceLbiNode* first_diff_node = nullptr;
  {
    const auto& first = node2mut_ref_ancestors.at(last_mut_ref_nodes.at(0));
    const auto& second = node2mut_ref_ancestors.at(last_mut_ref_nodes.at(1));
    first_diff_node = GetFirstDiffNode(first, second);
    if (first_diff_node == nullptr) {
      first_diff_node = last_mut_ref_nodes.at(first.size() < second.size() ? 0 : 1);
    }
  }
  return first_diff_node->GetSoleValidInEdge(IsValidEdge);
}

void InplaceLbiGraph::GetSafeInplaceObnEdges(
    const HashSet<const InplaceLbiNode*>& nodes,
    const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge,
    const std::function<bool(const LogicalBlobId&, const std::string&)>&
        IsLbiAllConsumerReachableToOpName,
    HashSet<const InplaceLbiEdge*>* cur_disabled_edges) const {
  ForEachTree(nodes, IsValidEdge, [&](const HashSet<const InplaceLbiNode*>& nodes) {
    // no inter-op reference conflicts
    const InplaceLbiEdge* inter_op_conflict_ref_edge = FindFirstInterOpRefConflictMutRefEdge(
        nodes, IsValidEdge, IsLbiAllConsumerReachableToOpName);
    // mutable reference always goes after const reference
    const InplaceLbiEdge* const_ref_conflict_ref_edge =
        FindFirstConstRefConflictMutRefEdge(nodes, IsValidEdge, IsLbiAllConsumerReachableToOpName);
    // no intra-op reference conflicts
    const InplaceLbiEdge* intra_op_conflict_ref_edge =
        FindFirstIntraOpRefConflictMutRefEdge(nodes, IsValidEdge);
    if (const_ref_conflict_ref_edge == nullptr && intra_op_conflict_ref_edge == nullptr
        && inter_op_conflict_ref_edge == nullptr) {
      FindAllEdges(nodes, IsValidEdge, cur_disabled_edges);
    }
  });
}

void InplaceLbiGraph::ForEachTree(
    const HashSet<const InplaceLbiNode*>& nodes,
    const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge,
    const std::function<void(const HashSet<const InplaceLbiNode*>&)>& Handler) const {
  auto ForEachNode = [&](const std::function<void(const InplaceLbiNode*)>& Handler) {
    for (const auto* node : nodes) { Handler(node); }
  };
  auto ForEachInNode = GetForEachValidInNode(&nodes, IsValidEdge);
  auto ForEachOutNode = GetForEachValidOutNode(&nodes, IsValidEdge);
  auto ForEachConnected = [&](const InplaceLbiNode* node,
                              const std::function<void(const InplaceLbiNode*)>& Handler) {
    ForEachInNode(node, Handler);
    ForEachOutNode(node, Handler);
  };
  ForEachConnectedComponent(ForEachNode, ForEachConnected, Handler);
}

void InplaceLbiGraph::FixConstRefOrMutRefConflictsToUpdtNode(
    const HashSet<const InplaceLbiNode*>& nodes,
    const std::function<bool(const LogicalBlobId&, const std::string&)>&
        IsLbiAllConsumerReachableToOpName,
    HashSet<const InplaceLbiEdge*>* cur_disabled_edges) const {
  auto IsValidEdge = [](const InplaceLbiEdge*) { return true; };
  const InplaceLbiNode* updt_node = nullptr;
  HashSet<const InplaceLbiNode*> safe_const_ref_nodes;
  const InplaceLbiNode* root = GetRoot(nodes, IsValidEdge);
  CHECK_NOTNULL(root);
  {
    const auto* source_op_root = dynamic_cast<const SourceOpInplaceLbiNode*>(root);
    CHECK_NOTNULL(source_op_root);
    updt_node = FindSoleIsMutableIbnConsumer(source_op_root);
    CHECK_NOTNULL(updt_node);
    auto ForEachNext = [&](const InplaceLbiNode* node,
                           const std::function<void(const InplaceLbiNode*)>& Handler) {
      node->ForEachNodeOnValidOutEdge(IsValidEdge, [&](const InplaceLbiNode* out_node) {
        if (dynamic_cast<const NormalInplaceLbiNode*>(out_node) == nullptr) { return; }
        if (out_node->IsMutRef(IsValidEdge)) { return; }
        if (!IsLbiAllConsumerReachableToOpName(out_node->lbi(), updt_node->lbi().op_name())) {
          return;
        }
        Handler(out_node);
      });
    };
    BfsForEachNode({root}, ForEachNext, [&](const InplaceLbiNode* node) {
      if (node == root) { return; }
      CHECK(safe_const_ref_nodes.emplace(node).second);
    });
  }
  for (const auto* node : safe_const_ref_nodes) {
    node->ForEachNodeOnValidOutEdge(IsValidEdge, [&](const InplaceLbiNode* out_node) {
      if (safe_const_ref_nodes.find(out_node) == safe_const_ref_nodes.end()
          && out_node != updt_node) {
        CHECK(nodes.find(out_node) != nodes.end());
        CHECK(cur_disabled_edges->emplace(out_node->GetSoleValidInEdge(IsValidEdge)).second);
      }
    });
  }
  // remove mutable inplace edges from root which are not end with model update node
  root->ForEachNodeOnValidOutEdge(IsValidEdge, [&](const InplaceLbiNode* out_node) {
    const auto* node = dynamic_cast<const NormalInplaceLbiNode*>(out_node);
    if (node != nullptr && node->IsMutRef(IsValidEdge)) {
      CHECK(nodes.find(out_node) != nodes.end());
      CHECK(cur_disabled_edges->emplace(node->GetSoleValidInEdge(IsValidEdge)).second);
    }
  });
}

void InplaceLbiGraph::FixMutRefConflictsFromSourceOpNode(
    const SourceOpInplaceLbiNode* root,
    const std::function<bool(const InplaceLbiEdge*)>& IsValidEdge,
    HashSet<const InplaceLbiEdge*>* cur_disabled_edges) const {
  HashSet<const InplaceLbiNode*> safe_const_ref_nodes;
  {
    auto ForEachNext = [&](const InplaceLbiNode* node,
                           const std::function<void(const InplaceLbiNode*)>& Handler) {
      node->ForEachNodeOnValidOutEdge(IsValidEdge, [&](const InplaceLbiNode* out_node) {
        if (dynamic_cast<const NormalInplaceLbiNode*>(out_node) == nullptr) {
          Handler(out_node);
        } else if (out_node->IsConstRef(IsValidEdge)) {
          Handler(out_node);
        } else {
          // do nothing
        }
      });
    };
    BfsForEachNode({root}, ForEachNext, [&](const InplaceLbiNode* node) {
      if (dynamic_cast<const NormalInplaceLbiNode*>(node) != nullptr) {
        CHECK(safe_const_ref_nodes.emplace(node).second);
      }
    });
  }
  for (const auto* node : safe_const_ref_nodes) {
    node->ForEachNodeOnValidOutEdge(IsValidEdge, [&](const InplaceLbiNode* out_node) {
      if (safe_const_ref_nodes.find(out_node) == safe_const_ref_nodes.end()
          && dynamic_cast<const NormalInplaceLbiNode*>(out_node) != nullptr
          && out_node->IsMutRef(IsValidEdge)) {
        CHECK(cur_disabled_edges->emplace(out_node->GetSoleValidInEdge(IsValidEdge)).second);
      }
    });
  }
}

}  // namespace oneflow
