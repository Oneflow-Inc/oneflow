#include <unordered_map>
#include <unordered_set>
#include "oneflow/core/compiler/of2xla/xla_graph.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/pass/xla_optimize_pass.h"

namespace oneflow {
namespace mola {

class ClusterCompiledOpPass : public XlaOptimizePass {
 public:
  ClusterCompiledOpPass(const OptimizeOptions &options)
      : XlaOptimizePass(options) {}

  void Run() override;

 private:
  class Cluster {
   public:
    Cluster() : id_(-1), nodes_(nullptr) {}
    explicit Cluster(int32_t id,
                     const std::unordered_set<const XlaNode *> *nodes)
        : id_(id), nodes_(nodes) {}
    // Merge other cluster with matched id
    void Merge(const Cluster &other);

    const std::unordered_set<const XlaNode *> &Nodes() const {
      return *nodes_;
    }

   private:
    int32_t id_;
    const std::unordered_set<const XlaNode *> *nodes_;
  };

  // void RerankClusterId();

  void UpdateClusterInfoToGraph(
      const std::unordered_map<int32_t, Cluster> &clusters,
      XlaGraph *graph);
};

void ClusterCompiledOpPass::Run() {
  XlaGraph *graph = new XlaGraph;
  graph->CopyFrom(*(this->optimize_options_.graph));

  std::unordered_map<int64_t, XlaNode *> compiled_nodes;
  std::unordered_set<int64_t> removed_nodes;

  int index = 0;
  for (auto &node : graph->Nodes()) {
    if (node->IsCompiled()) {
      node->set_cluster_id(index++);
      compiled_nodes.emplace(node->unique_id(), node);
    }
  }

  for (auto &kv : compiled_nodes) {
    if (removed_nodes.count(kv.first) != 0) {
      continue;
    }

    // node has not been moved from the graph
    XlaNode *node = kv.second;
    std::stack<const XlaEdge *> out_edges;
    for (const XlaEdge *edge : node->out_edges()) {
      out_edges.push(edge);
    }

    while (!out_edges.empty()) {
      const XlaEdge *edge = out_edges.top();
      out_edges.pop();

      DCHECK(edge->start() == node);
      const XlaNode *end = edge->end();
      // TODO(hjchen2) Sbp signatures and control flow should be token into
      // consideration
      if (end->IsCompiled() && compiled_nodes.count(end->unique_id())) {
        // Reset input edges's end node
        for (XlaEdge *in_edge : end->in_edges()) {
          if (in_edge->start() == node) {
            continue;
          }
          in_edge->UpdateEndNode(node);
        }
        // Reset output edges's start node
        for (XlaEdge *out_edge : end->out_edges()) {
          out_edges.push(out_edge);
          out_edge->UpdateStartNode(node);
        }
        node->fold(end);
        // remove the end node
        removed_nodes.insert(end->unique_id());
      }
    }
  }

  std::unordered_map<int32_t, Cluster> clusters;
  // filter invalid cluster
  int32_t minimum_nodes_in_cluster =
      this->optimize_options_.minimum_nodes_in_cluster;
  for (auto &kv : compiled_nodes) {
    XlaNode *node = kv.second;
    if (removed_nodes.count(kv.first) ||
        node->folded_nodes().size() < minimum_nodes_in_cluster) {
      continue;
    }
    int32_t cluster_id = node->cluster_id();
    Cluster cluster(cluster_id, &node->folded_nodes());
    clusters[cluster_id].Merge(cluster);
  }
  
  // Rerank cluster id start by 0
  // RerankClusterId(clusters);

  UpdateClusterInfoToGraph(clusters, this->optimize_options_.graph);

  // free the temporary graph
  delete graph;
}

void ClusterCompiledOpPass::UpdateClusterInfoToGraph(
    const std::unordered_map<int32_t, Cluster> &clusters,
    XlaGraph *graph) {
  for (const auto &kv : clusters) {
    int32_t cluster_id = kv.first;
    // Since the nodes in clusters are copied from the graph, so we should
    // find the original node by it's unique id
    for (const XlaNode *n : kv.second.Nodes()) {
      XlaNode *node = graph->Node(n->unique_id());
      node->set_cluster_id(cluster_id);
    }
  }
}

void ClusterCompiledOpPass::Cluster::Merge(
    const ClusterCompiledOpPass::Cluster &other) {
  DCHECK_EQ(id_, other.id_);
  auto *nodes = const_cast<std::unordered_set<const XlaNode *> *>(nodes_);
  nodes->insert(other.nodes_->begin(), other.nodes_->end());
}

REGISTER_OPTIMIZE_PASS(ClusterCompiledOp, ClusterCompiledOpPass);

}  // namespace mola
}  // namespace oneflow
