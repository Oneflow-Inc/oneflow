#include <unordered_map>
#include <unordered_set>
#include "oneflow/engine/pass/optimize_pass.h"
#include "oneflow/engine/xla/of2xla/xla_graph.h"
#include "oneflow/engine/xla/of2xla/xla_utility.h"

namespace oneflow {
namespace mola {

class MarkClusterIdPass : public OptimizePass {
 public:
  MarkClusterIdPass(const OptimizeOptions &options)
      : OptimizePass(options) {}

  void Run() override;

 private:
  class Cluster {
   public:
    Cluster() : id_(-1) {}
    explicit Cluster(int64_t id) : id_(id) {}
    virtual ~Cluster() { Clear(); }

    void Merge(const Cluster &other);
    bool InsertNode(const XlaNode *node);
    bool EraseNode(const XlaNode *node);
    void Clear();

    bool TryInsertNode(const XlaNode *node); 
    bool HasNode(const XlaNode *node) const {
      return nodes_.count(node) > 0;
    }

    int64_t id() const { return id_; }
    size_t size() const { return nodes_.size(); }
    const std::unordered_set<const XlaNode *> &nodes() const {
      return nodes_;
    }

   private:
    bool IsReachable(const Cluster &target);

    int64_t id_;
    std::unordered_set<const XlaNode *> nodes_;
    std::unordered_set<const XlaEdge *> input_edges_;
    std::unordered_set<const XlaEdge *> output_edges_;
  };

  bool SatisfySbpPolicy(const XlaEdge *edge) const;
  bool SatisfyTimeShape(const XlaEdge *edge) const;

  bool TryToFuseWithParentCluster(const XlaNode *node, int64_t cluster_id,
                                  const std::vector<XlaEdge *> &in_edges);

  // void RerankClusterId();

  void WriteClusterInfoToGraph(XlaGraph *graph);

  std::vector<Cluster> clusters_;
  std::unordered_map<XlaNode *, int64_t> node_cluster_ids_;
};

bool MarkClusterIdPass::Cluster::InsertNode(const XlaNode *node) {
  if (!nodes_.insert(node).second) {
    return false;
  }
  for (const XlaEdge *edge : node->in_edges()) {
    if (HasNode(edge->start())) {
      output_edges_.erase(edge);
    } else {
      input_edges_.insert(edge);
    }
  }
  for (const XlaEdge *edge : node->out_edges()) {
    if (HasNode(edge->end())) {
      input_edges_.erase(edge);
    } else {
      output_edges_.insert(edge);
    }
  }
  return true;
}

bool MarkClusterIdPass::Cluster::EraseNode(const XlaNode *node) {
  if (nodes_.erase(node) == 0) {
    return false;
  }
  for (const XlaEdge *edge : node->in_edges()) {
    if (HasNode(edge->start())) {
      output_edges_.insert(edge);
    }
  }
  for (const XlaEdge *edge : node->out_edges()) {
    if (HasNode(edge->end())) {
      input_edges_.insert(edge);
    }
  }
  return true;
}

void MarkClusterIdPass::Cluster::Clear() {
  nodes_.clear();
  input_edges_.clear();
  output_edges_.clear();
}

bool MarkClusterIdPass::Cluster::IsReachable(const Cluster &target) {
  std::unordered_set<const XlaNode *> visited_nodes;
  std::stack<const XlaNode *> stack;
  for (const XlaEdge *edge : output_edges_) {
    stack.push(edge->end());
  }

  while (!stack.empty()) {
    const XlaNode *node = stack.top();
    stack.pop();
    if (target.HasNode(node)) {
      return true;
    }
    for (const XlaEdge *edge : node->out_edges()) {
      const XlaNode *end = edge->end();
      if (visited_nodes.insert(end).second) {
        stack.push(end);
      }
    }
  }
  return false;
}

bool MarkClusterIdPass::Cluster::TryInsertNode(const XlaNode *node) {
  if (InsertNode(node) && IsReachable(*this)) {
    EraseNode(node);
    return false;
  }
  return true;
}

bool MarkClusterIdPass::SatisfySbpPolicy(const XlaEdge *edge) const {
  return this->optimize_options_.ignore_sbp_policy ||
         edge->IsControlEdge() ||
         (edge->sbp_policy(0) == edge->sbp_policy(1));
}

bool MarkClusterIdPass::SatisfyTimeShape(const XlaEdge *edge) const {
  return this->optimize_options_.ignore_time_shape ||
         edge->IsControlEdge() ||
         (edge->time_shape(0) == edge->time_shape(1));
}

bool MarkClusterIdPass::TryToFuseWithParentCluster(
    const XlaNode *node, int64_t cluster_id,
    const std::vector<XlaEdge *> &in_edges) {
  bool status = true;
  for (const XlaEdge *edge : in_edges) {
    status = status && edge->start()->backend() == node->backend() &&
             (edge->IsControlEdge() ||
             (SatisfySbpPolicy(edge) && SatisfyTimeShape(edge)));
  }

  Cluster &cluster = clusters_[cluster_id];
  if (status) {
    return cluster.TryInsertNode(node);
  }
  return false;
}

void MarkClusterIdPass::Run() {
  XlaGraph *graph = this->optimize_options_.graph;

  TopologyVisit(*graph, [this](XlaNode *node) -> void {
    if (!node->IsCompiled()) {
      return;
    }

    bool has_been_fused = [&]() -> bool {
      std::unordered_map<int64_t, std::vector<XlaEdge *>> candidate_edges;
      for (XlaEdge *edge : node->in_edges()) {
        XlaNode *parent = edge->start();
        if (node_cluster_ids_.count(parent)) {
          int64_t cluster_id = node_cluster_ids_[parent];
          candidate_edges[cluster_id].push_back(edge);
        }
      }

      for (const auto &it : candidate_edges) {
        int64_t cluster_id = it.first;
        const std::vector<XlaEdge *> &edges = it.second;
        if (TryToFuseWithParentCluster(node, cluster_id, edges)) {
          node_cluster_ids_[node] = cluster_id;
          return true;
        }
      }
      return false;
    }();

    if (!has_been_fused) {
      clusters_.emplace_back(clusters_.size());
      Cluster &cluster = clusters_.back();
      cluster.InsertNode(node);
      node_cluster_ids_[node] = cluster.id();
    }
  });

  // TODO(hjchen2) merge clusters

  // Clear invalid cluster
  int32_t minimum_nodes_in_cluster =
      this->optimize_options_.minimum_nodes_in_cluster;
  int32_t maximum_nodes_in_cluster =
      this->optimize_options_.maximum_nodes_in_cluster;
  for (Cluster &cluster : clusters_) {
    if (cluster.size() < minimum_nodes_in_cluster ||
        cluster.size() > maximum_nodes_in_cluster) {
      cluster.Clear();
    }
  }
  // Rerank cluster id start by 0
  // RerankClusterId(clusters);

  WriteClusterInfoToGraph(this->optimize_options_.graph);
}

void MarkClusterIdPass::WriteClusterInfoToGraph(XlaGraph *graph) {
  for (const Cluster &cluster : clusters_) {
    for (const XlaNode *n : cluster.nodes()) {
      XlaNode *node = graph->Node(n->unique_id());
      node->set_cluster_id(cluster.id());
    }
  }
}

void MarkClusterIdPass::Cluster::Merge(
    const MarkClusterIdPass::Cluster &other) {
  DCHECK_EQ(id_, other.id_);
  nodes_.insert(other.nodes_.begin(), other.nodes_.end());
}

REGISTER_OPTIMIZE_PASS(MarkClusterId, MarkClusterIdPass);

}  // namespace mola
}  // namespace oneflow
