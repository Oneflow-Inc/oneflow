#include <unordered_map>
#include <unordered_set>
#include "oneflow/xla/of2xla/xla_graph.h"
#include "oneflow/xla/of2xla/xla_utility.h"
#include "oneflow/xla/of2xla/pass/cluster.h"
#include "oneflow/xla/of2xla/pass/xla_optimize_pass.h"

namespace oneflow {
namespace mola {

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

  util::Set<ClusterNode *> FindAllParents(ClusterNode *node) const;
  util::Set<ClusterNode *> FindAllChildrens(ClusterNode *node) const;
  void DisableExtraPreDependence(const std::string &type);
  void DisableExtraPostDependence(const std::string &type);

  bool TryToFuseWithParent(ClusterNode *children, ClusterNode *parent);

  void BuildClusterNodesAndEdges();

  void DetermineFusionDisabledEdges();

  void AddNoNodeForBetterClustering();

  void ClusteringSubgraphs();

  void RemoveInvalidClusterNodes();

  // Rerank cluster id start by 0
  void RerankClusterIds();

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

  bool strict_clustering = this->optimize_options_.strict_clustering;
  if (strict_clustering) {
    for (const ClusterEdge *edge : parent->out_edges()) {
      if (edge->end() != children &&
          !IsNodeDirectChildren(children, edge->end())) {
        can_fusion = false;
      }
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
    for (const XlaEdge *edge : xrt_node->out_edges()) {
      int64_t unique_id = edge->end()->unique_id();
      ClusterNode *end = cluster_nodes[unique_id];
      auto cluster_edge = BuildClusterEdge(start, end);
      cluster_edge->set_is_control_edge(edge->IsControlEdge());
      cluster_edge->set_start_sbp_policy(edge->sbp_policy(0));
      cluster_edge->set_end_sbp_policy(edge->sbp_policy(1));
      cluster_edge->set_start_time_shape(edge->time_shape(0));
      cluster_edge->set_end_time_shape(edge->time_shape(1));  
      start->AddOutEdge(cluster_edge.get());
      end->AddInEdge(cluster_edge.get());
      allocated_edges_.push_back(cluster_edge);
    }
  }
}

util::Set<ClusterNode *> MarkClusterIdPass::FindAllParents(
    ClusterNode *node) const {
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

util::Set<ClusterNode *> MarkClusterIdPass::FindAllChildrens(
    ClusterNode *node) const {
  std::unordered_set<ClusterNode *> visited;
  std::queue<ClusterNode *> visit_queue;
  visit_queue.push(node);

  while (!visit_queue.empty()) {
    ClusterNode *n = visit_queue.front();
    visit_queue.pop();
    for (ClusterEdge *edge : n->out_edges()) {
      ClusterNode *p = edge->end();
      if (visited.insert(p).second) {
        visit_queue.push(p);
      }
    }
  }
  return std::move(visited);
}

void MarkClusterIdPass::DisableExtraPreDependence(const std::string &type) {
  std::vector<ClusterNode *> reference_nodes;
  for (ClusterNode *node : Nodes()) {
    if (node->type() == type) {
      reference_nodes.push_back(node);
    }
  }

  for (ClusterNode *node : reference_nodes) {
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

void MarkClusterIdPass::DisableExtraPostDependence(const std::string &type) {
  std::vector<ClusterNode *> reference_nodes;
  for (ClusterNode *node : Nodes()) {
    if (node->type() == type) {
      reference_nodes.push_back(node);
    }
  }

  for (ClusterNode *node : reference_nodes) {
    util::Set<ClusterNode *> childrens = FindAllChildrens(node);
    for (ClusterNode *p : childrens) {
      for (ClusterEdge *e : p->in_edges()) {
        if (!childrens.count(e->start())) {
          e->set_is_fusion_disabled(true);
        }
      }
    }
  }
}

void MarkClusterIdPass::DetermineFusionDisabledEdges() {
  DisableExtraPreDependence("ReduceConcat");
  DisableExtraPostDependence("ReduceSplit");
}

template <>
struct GraphTrait<MarkClusterIdPass> {
  typedef ClusterNode *pNodeType;
  typedef ClusterEdge *pEdgeType;
};

void MarkClusterIdPass::ClusteringSubgraphs() {
  int32_t maximum_nodes =
      this->optimize_options_.clustering_maximum_nodes;
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

    // for (ClusterNode *node : ordered_nodes) {
    for (int i = ordered_nodes.size() - 1; i >= 0; --i) {
      ClusterNode *node = ordered_nodes[i];
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
}

void MarkClusterIdPass::AddNoNodeForBetterClustering() {
  auto BuildControlEdge = [](ClusterNode *start, ClusterNode *end)
    -> std::shared_ptr<ClusterEdge> {
    auto cluster_edge = std::make_shared<ClusterEdge>(start, end);
    cluster_edge->set_is_control_edge(true);
    return std::move(cluster_edge);
  };

  for (ClusterNode *start : root_nodes_) {
    util::Set<ClusterNode *> childrens;
    if (start->type() == "ReduceSplit") {
      for (ClusterEdge *edge : start->out_edges()) {
        childrens.insert(edge->end());
      }
    }
    int64_t cluster_id = allocated_nodes_.size();
    auto no_node = std::make_shared<NoClusterNode>(cluster_id);
    no_node->set_backend(start->backend());
    allocated_nodes_.push_back(no_node);
    root_nodes_.emplace(no_node.get());
    for (ClusterNode *child : childrens) {
      auto control_edge = BuildControlEdge(no_node.get(), child);
      no_node->AddOutEdge(control_edge.get());
      child->AddInEdge(control_edge.get());
      allocated_edges_.push_back(control_edge);
    }
  }
}

void MarkClusterIdPass::RemoveInvalidClusterNodes() {
  int32_t minimum_nodes =
      this->optimize_options_.clustering_minimum_nodes;
  int32_t maximum_nodes =
      this->optimize_options_.clustering_maximum_nodes;

  std::vector<ClusterNode *> removing_clusters;
  for (ClusterNode *node : root_nodes_) {
    if (!node->IsCompiled() || node->size() < minimum_nodes ||
        node->size() > maximum_nodes ||
        (IsNoClusterNode(node) && node->size() == 1)) {
      removing_clusters.push_back(node);
    }
  }
  for (ClusterNode *node : removing_clusters) {
    root_nodes_.erase(node);
  }
}

void MarkClusterIdPass::RerankClusterIds() {
  int64_t rank = 0;
  for (ClusterNode *node : root_nodes_) {
    if (!(IsNoClusterNode(node) && node->size() == 1)) {
      node->set_id(rank++);
    }
  }
}

void MarkClusterIdPass::WriteClusterInfoToGraph(XlaGraph *graph) {
  for (const ClusterNode *node : root_nodes_) {
    for (const ClusterNode *folded_node : node->folded_nodes()) {
      if (!IsNoClusterNode(folded_node)) {
        int64_t unique_id = folded_node->xrt_node()->unique_id();
        XlaNode *xrt_node = graph->Node(unique_id);
        xrt_node->set_cluster_id(node->id());
      }
    }
  }
}

void MarkClusterIdPass::Run() {
  CHECK(allocated_nodes_.empty());

  BuildClusterNodesAndEdges();

  DetermineFusionDisabledEdges();

  AddNoNodeForBetterClustering();

  // Clustering nodes iteratively
  ClusteringSubgraphs();

  RemoveInvalidClusterNodes();

  RerankClusterIds();

  WriteClusterInfoToGraph(this->optimize_options_.graph);
}

REGISTER_OPTIMIZE_PASS(MarkClusterId, MarkClusterIdPass);

}  // namespace mola
}  // namespace oneflow
