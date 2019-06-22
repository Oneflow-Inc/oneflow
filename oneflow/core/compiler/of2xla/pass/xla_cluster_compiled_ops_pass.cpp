#include <unordered_map>
#include <unordered_set>
#include "oneflow/core/compiler/of2xla/xla_graph.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/pass/xla_optimize_pass.h"

namespace oneflow {
namespace mola {

class ClusterCompiledOpsPass : public XlaOptimizePass {
 public:
  ClusterCompiledOpsPass(const OptimizeOptions &options)
      : XlaOptimizePass(options) {}

  void Run() override;

 private:
  class Cluster {
   public:
    Cluster() : id_(-1) {}
    explicit Cluster(int64_t id) : id_(id) {}
    // Merge other cluster with matched id
    void merge(const Cluster &other);
    void push(int64_t node) { nodes_.insert(node); }
    void remove(int64_t node) { nodes_.erase(node); }
    void clear() { nodes_.clear(); }
    
    int64_t id() const { return id_; }
    size_t size() const { return nodes_.size(); }

    const std::unordered_set<int64_t> &nodes() const {
      return nodes_;
    }

   private:
    int64_t id_;
    std::unordered_set<int64_t> nodes_;
  };

  void InfectNeighborNodes(XlaNode *node,
                           std::unordered_set<int64_t> *visited_nodes,
                           Cluster *cluster);
  // void RerankClusterId();

  void UpdateClusterInfoToGraph(const std::vector<Cluster> &clusters,
                                XlaGraph *graph);
};

void ClusterCompiledOpsPass::Run() {
  XlaGraph *graph = this->optimize_options_.graph;

  std::unordered_set<int64_t> visited_nodes;
  std::vector<Cluster> clusters;

  for (auto &node : graph->Nodes()) {
    int64_t unique_id = node->unique_id();
    if (node->IsCompiled() && (visited_nodes.count(unique_id) == 0)) {
      visited_nodes.insert(unique_id);

      clusters.emplace_back(clusters.size());
      Cluster &cluster = clusters.back();
      cluster.push(unique_id);

      InfectNeighborNodes(node, &visited_nodes, &cluster);
    }
  }

  // filter invalid cluster
  int32_t minimum_nodes_in_cluster =
      this->optimize_options_.minimum_nodes_in_cluster;
  for (auto &cluster : clusters) {
    if (cluster.size() < minimum_nodes_in_cluster) {
      cluster.clear();
    }
  }
  // Rerank cluster id start by 0
  // RerankClusterId(clusters);

  UpdateClusterInfoToGraph(clusters, this->optimize_options_.graph);
}

void ClusterCompiledOpsPass::InfectNeighborNodes(
    XlaNode *node, std::unordered_set<int64_t> *visited_nodes,
    Cluster *cluster) {
  auto visited_fn = [&](XlaNode *n) -> bool {
    int64_t unique_id = n->unique_id();
    return (visited_nodes->count(unique_id) != 0);
  };

  std::stack<XlaNode *> stack;
  stack.push(node);

  while (!stack.empty()) {
    XlaNode *top_node = stack.top();
    stack.pop();

    // Visit input edges, and try to infect front nodes
    for (auto &e : top_node->in_edges()) {
      XlaNode *front = e->start();
      // TODO(hjchen2) Sbp signatures and control flow should be
      // token into consideration
      if (front->IsCompiled() && !visited_fn(front)) {
        visited_nodes->insert(front->unique_id());
        cluster->push(front->unique_id());
        stack.push(front);
      }
    }
    // Visit output edges, and try to infect latter nodes
    for (auto &e : top_node->out_edges()) {
      XlaNode *latter = e->end();
      // TODO(hjchen2) Sbp signatures and control flow should be
      // token into consideration
      if (latter->IsCompiled() && !visited_fn(latter)) {
        visited_nodes->insert(latter->unique_id());
        cluster->push(latter->unique_id());
        stack.push(latter);
      }
    }
  }
}

void ClusterCompiledOpsPass::UpdateClusterInfoToGraph(
    const std::vector<Cluster> &clusters, XlaGraph *graph) {
  for (const auto &cluster : clusters) {
    for (int64_t node_id : cluster.nodes()) {
      XlaNode *node = graph->Node(node_id);
      node->set_cluster_id(cluster.id());
    }
  }
}

void ClusterCompiledOpsPass::Cluster::merge(
    const ClusterCompiledOpsPass::Cluster &other) {
  DCHECK_EQ(id_, other.id_);
  nodes_.insert(other.nodes_.begin(), other.nodes_.end());
}

REGISTER_OPTIMIZE_PASS(ClusterCompiledOps, ClusterCompiledOpsPass);

}  // namespace mola
}  // namespace oneflow
