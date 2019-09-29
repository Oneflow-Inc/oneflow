#include <unordered_map>
#include <unordered_set>
#include "oneflow/xla/of2xla/xla_graph.h"
#include "oneflow/xla/of2xla/xla_utility.h"
#include "oneflow/xla/of2xla/pass/xla_optimize_pass.h"

namespace oneflow {
namespace mola {

namespace util {
template <typename K, typename T>
using Map = std::unordered_map<K, T>;
template <typename T>
using Set = std::unordered_set<T>;
}  // namespace util

class ClusterNode;
class ClusterEdge {
 public:
  ClusterEdge() = default;
  ClusterEdge(ClusterNode *start, ClusterNode *end)
      : start_(start), end_(end) {}
  virtual ~ClusterEdge() {}

  void UpdateStartNode(ClusterNode *start) { start_ = start; }
  void UpdateEndNode(ClusterNode *end) { end_ = end; }

  ClusterNode *start() const { return start_; }
  ClusterNode *end() const { return end_; }

  bool is_control_edge() const { return is_control_edge_; }
  void set_is_control_edge(bool is_control_edge) {
    is_control_edge_ = is_control_edge;
  }

  bool is_fusion_disabled() const { return is_fusion_disabled_; }
  void set_is_fusion_disabled(bool is_fusion_disabled) {
    is_fusion_disabled_ = is_fusion_disabled;
  }

  bool IsIdentity() const {
    return start_sbp_policy() == end_sbp_policy();
  }

  SbpParallel start_sbp_policy() const { return sbp_policy_[0]; }
  SbpParallel end_sbp_policy() const { return sbp_policy_[1]; }
  void set_start_sbp_policy(const SbpParallel &sbp_policy) {
    sbp_policy_[0] = sbp_policy;
  }
  void set_end_sbp_policy(const SbpParallel &sbp_policy) {
    sbp_policy_[1] = sbp_policy;
  }
  Shape start_time_shape() const { return time_shape_[0]; }
  Shape end_time_shape() const { return time_shape_[1]; }
  void set_start_time_shape(const Shape &shape) { time_shape_[0] = shape; }
  void set_end_time_shape(const Shape &shape) { time_shape_[1] = shape; }
  
 protected:
  ClusterNode *start_;
  ClusterNode *end_;
  SbpParallel sbp_policy_[2];
  Shape time_shape_[2];
  bool is_control_edge_ = false;
  bool is_fusion_disabled_ = false;
};

class ClusterNode {
 public:
  ClusterNode() : ClusterNode(nullptr, -1) {}
  explicit ClusterNode(int64_t id) : ClusterNode(nullptr, id) {}
  explicit ClusterNode(const XlaNode *node, int64_t id)
      : xrt_node_(node), id_(id) {
    folded_nodes_.insert(this);
  }
  virtual ~ClusterNode() {}

  util::Set<ClusterEdge *> &in_edges() { return in_edges_; }
  util::Set<ClusterEdge *> &out_edges() { return out_edges_; }
  const util::Set<ClusterEdge *> &in_edges() const { return in_edges_; }
  const util::Set<ClusterEdge *> &out_edges() const { return out_edges_; }

  void AddInEdge(const ClusterEdge *edge);
  void AddOutEdge(const ClusterEdge *edge);
  void EraseInEdge(const ClusterEdge *edge);
  void EraseOutEdge(const ClusterEdge *edge);
  void ClearInEdges() { in_edges_.clear(); }
  void ClearOutEdges() { out_edges_.clear(); }
  
  void Merge(ClusterNode &other);
  bool TryMerge(ClusterNode &other);
  bool IsReachable(const ClusterNode &target);
  bool IsSatisfySbpPolicy() const;
  bool IsSourceNode() const { return in_edges_.empty(); }
  bool IsCompiled() const { return xrt_node_->IsCompiled(); }

  bool operator==(const ClusterNode &other) const {
    return id_ == other.id_;
  }

  const XlaNode *xrt_node() const { return xrt_node_; }
  int64_t id() const { return id_; }
  void set_id(int64_t id) { id_ = id; }
  std::string type() const { return xrt_node_->op_type(); }
  std::string name() const { return xrt_node_->op_name(); }
  std::string backend() const { return xrt_node_->backend(); }
  size_t size() const { return folded_nodes_.size(); }
  const util::Set<ClusterNode *> &folded_nodes() const {
    return folded_nodes_;
  }
  util::Set<ClusterNode *> &folded_nodes() { return folded_nodes_; }

 protected:
  const XlaNode *xrt_node_;
  int64_t id_;
  util::Set<ClusterNode *> folded_nodes_;
  util::Set<ClusterEdge *> in_edges_;
  util::Set<ClusterEdge *> out_edges_;
};

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
  // TODO(hjchen2) Fix ReduceSplit
  util::Map<int64_t, util::Set<ClusterEdge *>> parent_edges;
  for (ClusterEdge *edge : in_edges_) {
    int64_t parent_id = edge->start()->id();
    parent_edges[parent_id].insert(edge);
    if (absl::StartsWith(edge->start()->name(),
            "System-Boxing-AllReduce-ReduceSplit") &&
        parent_edges[parent_id].size() > 1) {
      return false;
    }
  }

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
    for (const EdgeSnapshot &snapshot : involved_edges_) {
      snapshot.edge->UpdateStartNode(snapshot.from);
      snapshot.edge->UpdateEndNode(snapshot.to);
    }
    involved_edges_.clear();
  }

  void Complete() {
    Fallback();
    lhs_->Merge(*rhs_);
  }

 private:
  void BuildInputEdges() {
    for (ClusterEdge *edge : lhs_->in_edges()) {
      if (edge->start() != rhs_) {
        InvolveEdge(edge);
        edge->UpdateEndNode(this);
        AddInEdge(edge);
      }
    }
    for (ClusterEdge *edge : rhs_->in_edges()) {
      if (edge->start() != lhs_) {
        InvolveEdge(edge);
        edge->UpdateEndNode(this);
        AddInEdge(edge);
      }
    }
  }

  void BuildOutputEdges() {
    for (ClusterEdge *edge : lhs_->out_edges()) {
      if (edge->end() != rhs_) {
        InvolveEdge(edge);
        edge->UpdateStartNode(this);
        AddOutEdge(edge);
      }
    }
    for (ClusterEdge *edge : rhs_->out_edges()) {
      if (edge->end() != lhs_) {
        InvolveEdge(edge);
        edge->UpdateStartNode(this);
        AddOutEdge(edge);
      }
    }
  }

  void InvolveEdge(ClusterEdge *edge) {
    EdgeSnapshot snapshot;
    snapshot.from = edge->start();
    snapshot.to = edge->end();
    snapshot.edge = edge;
    involved_edges_.push_back(snapshot);
  }

  ClusterNode *lhs_;
  ClusterNode *rhs_;
  std::vector<EdgeSnapshot> involved_edges_;
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

  const util::Set<ClusterNode *> &nodes = other.folded_nodes();
  folded_nodes_.insert(nodes.begin(), nodes.end());
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

class MarkClusterIdPass : public XlaOptimizePass {
 public:
  MarkClusterIdPass(const OptimizeOptions &options)
      : XlaOptimizePass(options) {}

  void Run() override;

  // Add for `TopologyVisit`
  const util::Set<ClusterNode *> &Nodes() const {
    return root_nodes_;
  }

 private:
  bool IsSatisfyBackend(const ClusterEdge *edge) const;
  bool IsSatisfySbpPolicy(const ClusterEdge *edge) const;
  bool IsSatisfyTimeShape(const ClusterEdge *edge) const;

  util::Set<ClusterNode *> FindAllParents(ClusterNode *node);

  bool TryToFuseWithParent(ClusterNode *children, ClusterNode *parent);

  void BuildClusterNodesAndEdges();

  void DetermineFusionDisabledEdges();

  void RemoveInvalidClusterNodes();

  // Rerank cluster id start by 0
  void RerankClusterNodeIds();

  void WriteClusterInfoToGraph(XlaGraph *graph);

  // Root cluster nodes
  util::Set<ClusterNode *> root_nodes_;

  // All allocated nodes and edges which will always alive when
  // running the pass `MarkClusterIdPass`
  std::vector<std::shared_ptr<ClusterNode>> allocated_nodes_;
  std::vector<std::shared_ptr<ClusterEdge>> allocated_edges_;
};

bool MarkClusterIdPass::IsSatisfyBackend(const ClusterEdge *edge) const {
  return edge->start()->backend() == edge->end()->backend();
}

bool MarkClusterIdPass::IsSatisfySbpPolicy(const ClusterEdge *edge) const {
  return this->optimize_options_.ignore_sbp_policy ||
         edge->is_control_edge() ||
         (edge->start_sbp_policy() == edge->end_sbp_policy());
}

bool MarkClusterIdPass::IsSatisfyTimeShape(const ClusterEdge *edge) const {
  return this->optimize_options_.ignore_time_shape ||
         edge->is_control_edge() ||
         (edge->start_time_shape() == edge->end_time_shape());
}

bool MarkClusterIdPass::TryToFuseWithParent(ClusterNode *children,
                                            ClusterNode *parent) {
  bool can_fusion = true;
  for (const ClusterEdge *edge : children->in_edges()) {
    if (edge->start() == parent) {
      can_fusion = can_fusion && !edge->is_fusion_disabled() &&
                   IsSatisfyBackend(edge) && IsSatisfySbpPolicy(edge) &&
                   IsSatisfyTimeShape(edge);
    }
  }

  if (can_fusion) {
    return parent->TryMerge(*children);
  }
  return false;
}

void MarkClusterIdPass::BuildClusterNodesAndEdges() {
  XlaGraph *graph = this->optimize_options_.graph;
  util::Map<int64_t, ClusterNode *> cluster_nodes;

  TopologyVisit(*graph, [&](XlaNode *node) -> void {
    int64_t cluster_id = allocated_nodes_.size();
    auto cluster_node = std::make_shared<ClusterNode>(node, cluster_id);
    allocated_nodes_.push_back(cluster_node);
    root_nodes_.emplace(cluster_node.get());
    cluster_nodes.emplace(node->unique_id(), cluster_node.get());
  });

  auto BuildClusterEdge = [](ClusterNode *start, ClusterNode *end)
      -> std::shared_ptr<ClusterEdge> {
    return std::make_shared<ClusterEdge>(start, end);
  };
  for (ClusterNode *start : root_nodes_) {
    const XlaNode *xrt_node = start->xrt_node();
    for (const XlaEdge *xla_edge : xrt_node->out_edges()) {
      int64_t unique_id = xla_edge->end()->unique_id();
      ClusterNode *end = cluster_nodes[unique_id];
      auto cluster_edge = BuildClusterEdge(start, end);
      cluster_edge->set_is_control_edge(xla_edge->IsControlEdge());
      cluster_edge->set_start_sbp_policy(xla_edge->sbp_policy(0));
      cluster_edge->set_end_sbp_policy(xla_edge->sbp_policy(1));
      cluster_edge->set_start_time_shape(xla_edge->time_shape(0));
      cluster_edge->set_end_time_shape(xla_edge->time_shape(1));  
      start->AddOutEdge(cluster_edge.get());
      end->AddInEdge(cluster_edge.get());
      allocated_edges_.push_back(cluster_edge);
    }
  }
}

util::Set<ClusterNode *> MarkClusterIdPass::FindAllParents(ClusterNode *node) {
  std::unordered_set<ClusterNode *> visited;
  std::queue<ClusterNode *> visit_queue;
  visit_queue.push(node);

  while (!visit_queue.empty()) {
    ClusterNode *n = visit_queue.front();
    visit_queue.pop();
    for (ClusterEdge *edge : n->in_edges()) {
      ClusterNode *p = edge->start();
      if (visited.insert(p).second) {
        visit_queue.push(p);
      }
    }
  }
  return std::move(visited);
}

void MarkClusterIdPass::DetermineFusionDisabledEdges() {
  util::Set<std::string> io_types{"ReduceConcat"};
  std::vector<ClusterNode *> io_nodes;
  for (ClusterNode *node : Nodes()) {
    if (io_types.count(node->type())) {
      io_nodes.push_back(node);
    }
  }

  for (ClusterNode *node : io_nodes) {
    util::Set<ClusterNode *> parents = FindAllParents(node);
    for (ClusterNode *p : parents) {
      for (ClusterEdge *e : p->out_edges()) {
        if (!parents.count(e->end())) {
          e->set_is_fusion_disabled(true);
        }
      }
    }
  }
}

template <>
struct GraphTrait<MarkClusterIdPass> {
  typedef ClusterNode *pNodeType;
  typedef ClusterEdge *pEdgeType;
};

void MarkClusterIdPass::Run() {
  int32_t maximum_nodes =
      this->optimize_options_.maximum_nodes_in_cluster;

  CHECK(allocated_nodes_.empty());

  BuildClusterNodesAndEdges();

  DetermineFusionDisabledEdges();

  int32_t iter_count = 10;
  for (int i = 0; i < iter_count; ++i) {
    bool has_changed = false;
    std::vector<ClusterNode *> ordered_nodes;
    TopologyVisit(*this, [&](ClusterNode *node) {
      if (!node->IsCompiled()) {
        return;
      }
      ordered_nodes.push_back(node);
    });

    for (ClusterNode *node : ordered_nodes) {
      util::Set<ClusterNode *> candidate_parents;
      for (ClusterEdge *edge : node->in_edges()) {
        candidate_parents.insert(edge->start());
      }

      for (ClusterNode *parent : candidate_parents) {
        if (parent->IsCompiled() &&
            (parent->size() + node->size()) <= maximum_nodes &&
            TryToFuseWithParent(node, parent)) {
          has_changed = true;
          root_nodes_.erase(node);
          break;
        }
      }
    }
    if (!has_changed) {
      break;
    }
  }

  RemoveInvalidClusterNodes();

  RerankClusterNodeIds();

  WriteClusterInfoToGraph(this->optimize_options_.graph);
}

void MarkClusterIdPass::RemoveInvalidClusterNodes() {
  int32_t minimum_nodes =
      this->optimize_options_.minimum_nodes_in_cluster;
  int32_t maximum_nodes =
      this->optimize_options_.maximum_nodes_in_cluster;

  std::vector<ClusterNode *> removing_clusters;
  for (ClusterNode *node : root_nodes_) {
    if (!node->IsCompiled() || node->size() < minimum_nodes ||
        node->size() > maximum_nodes) {
      removing_clusters.push_back(node);
    }
  }
  for (ClusterNode *node : removing_clusters) {
    root_nodes_.erase(node);
  }
}

void MarkClusterIdPass::RerankClusterNodeIds() {
  int64_t rank = 0;
  for (ClusterNode *node : root_nodes_) {
    node->set_id(rank++);
  }
}

void MarkClusterIdPass::WriteClusterInfoToGraph(XlaGraph *graph) {
  for (const ClusterNode *node : root_nodes_) {
    for (const ClusterNode *folded_node : node->folded_nodes()) {
      int64_t unique_id = folded_node->xrt_node()->unique_id();
      XlaNode *xrt_node = graph->Node(unique_id);
      xrt_node->set_cluster_id(node->id());
    }
  }
}

REGISTER_OPTIMIZE_PASS(MarkClusterId, MarkClusterIdPass);

}  // namespace mola
}  // namespace oneflow
