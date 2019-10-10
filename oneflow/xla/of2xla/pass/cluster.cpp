#include "oneflow/xla/of2xla/pass/cluster.h"

namespace oneflow {
namespace mola {

void ClusterNode::AddInEdge(const ClusterEdge *edge) {
  in_edges_.insert(const_cast<ClusterEdge *>(edge));
}

void ClusterNode::AddOutEdge(const ClusterEdge *edge) {
  out_edges_.insert(const_cast<ClusterEdge *>(edge));
}

void ClusterNode::EraseInEdge(const ClusterEdge *edge) {
  in_edges_.erase(const_cast<ClusterEdge *>(edge));
}

void ClusterNode::EraseOutEdge(const ClusterEdge *edge) {
  out_edges_.erase(const_cast<ClusterEdge *>(edge));
}

bool ClusterNode::IsSatisfySbpPolicy() const {
  util::Map<int64_t, util::Set<bool>> connection_kinds;
  for (ClusterEdge *edge : in_edges_) {
    int64_t parent_id = edge->start()->id();
    connection_kinds[parent_id].insert(edge->IsIdentity());
    if (connection_kinds[parent_id].size() > 1) {
      return false;
    }
  }
  for (ClusterEdge *edge : out_edges_) {
    int64_t children_id = edge->end()->id();
    connection_kinds[children_id].insert(edge->IsIdentity());
    if (connection_kinds[children_id].size() > 1) {
      return false;
    }
  }
  return true;
}

bool ClusterNode::IsReachable(const ClusterNode &target) {
  util::Set<const ClusterNode *> visited_nodes;
  std::stack<const ClusterNode *> stack;
  for (const ClusterEdge *edge : out_edges_) {
    stack.push(edge->end());
  }

  while (!stack.empty()) {
    const ClusterNode *node = stack.top();
    stack.pop();
    if (target == *node) {
      return true;
    }
    for (const ClusterEdge *edge : node->out_edges()) {
      const ClusterNode *end = edge->end();
      if (visited_nodes.insert(end).second) {
        stack.push(end);
      }
    }
  }
  return false;
}

class ClusterMergeNode : public ClusterNode {
 public:
  struct EdgeSnapshot {
    ClusterNode *from;
    ClusterNode *to;
    ClusterEdge *edge;
  };

  ClusterMergeNode(ClusterNode *lhs, ClusterNode *rhs)
      : ClusterNode(-100), lhs_(lhs), rhs_(rhs) {
    BuildInputEdges();
    BuildOutputEdges();
  }

  virtual ~ClusterMergeNode() { Fallback(); }

  void Fallback() {
    for (const EdgeSnapshot &snapshot : snapshot_edges_) {
      snapshot.edge->UpdateStartNode(snapshot.from);
      snapshot.edge->UpdateEndNode(snapshot.to);
    }
    snapshot_edges_.clear();
  }

  void Complete() {
    Fallback();
    lhs_->Merge(*rhs_);
  }

 private:
  void BuildInputEdges() {
    for (ClusterEdge *edge : lhs_->in_edges()) {
      if (edge->start() != rhs_) {
        SnapshotEdge(edge);
        edge->UpdateEndNode(this);
        AddInEdge(edge);
      }
    }
    for (ClusterEdge *edge : rhs_->in_edges()) {
      if (edge->start() != lhs_) {
        SnapshotEdge(edge);
        edge->UpdateEndNode(this);
        AddInEdge(edge);
      }
    }
  }

  void BuildOutputEdges() {
    for (ClusterEdge *edge : lhs_->out_edges()) {
      if (edge->end() != rhs_) {
        SnapshotEdge(edge);
        edge->UpdateStartNode(this);
        AddOutEdge(edge);
      }
    }
    for (ClusterEdge *edge : rhs_->out_edges()) {
      if (edge->end() != lhs_) {
        SnapshotEdge(edge);
        edge->UpdateStartNode(this);
        AddOutEdge(edge);
      }
    }
  }

  void SnapshotEdge(ClusterEdge *edge) {
    EdgeSnapshot snapshot;
    snapshot.from = edge->start();
    snapshot.to = edge->end();
    snapshot.edge = edge;
    snapshot_edges_.push_back(snapshot);
  }

  ClusterNode *lhs_;
  ClusterNode *rhs_;
  std::vector<EdgeSnapshot> snapshot_edges_;
};

void ClusterNode::Merge(ClusterNode &other) {
  for (ClusterEdge *edge : other.in_edges()) {
    if (edge->start() != this) {
      edge->UpdateEndNode(this);
      AddInEdge(edge);
    } else {
      EraseOutEdge(edge);
    }
  }
  for (ClusterEdge *edge : other.out_edges()) {
    if (edge->end() != this) {
      edge->UpdateStartNode(this);
      AddOutEdge(edge);
    } else {
      EraseInEdge(edge);
    }
  }

  FoldNodes(other.folded_nodes());
}

bool ClusterNode::TryMerge(ClusterNode &other) {
  ClusterMergeNode node(this, &other);
  if (!node.IsSatisfySbpPolicy() || node.IsReachable(node)) {
    // Explicit fallback
    node.Fallback();
    return false;
  }
  node.Complete();
  return true;
}

bool IsNodeDirectChildren(const ClusterNode *parent,
                          const ClusterNode *children) {
  for (const ClusterEdge *edge : parent->out_edges()) {
    if (edge->end() == children) {
      return true;
    }
  }
  return false;
}

bool IsNoClusterNode(const ClusterNode *node) {
  return node->type() == "NoClusterNode";
}

}  // namespace mola
}  // namespace oneflow
