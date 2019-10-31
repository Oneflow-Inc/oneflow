#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/passes/cluster.h"
#include "oneflow/xrt/passes/pass.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {

bool IsSatisfyBackend(const ClusterEdge *edge) {
  return edge->start()->device() == edge->end()->device();
}
bool IsSatisfySbpPolicy(const ClusterEdge *edge) {
  return edge->is_control_edge() ||
         (edge->start_sbp_policy() == edge->end_sbp_policy());
}
bool IsSatisfyTimeShape(const ClusterEdge *edge) {
  return edge->is_control_edge() ||
         (edge->start_time_shape() == edge->end_time_shape());
}

class MarkClusterIdPass : public XrtPass {
 public:
  MarkClusterIdPass() = default;

  void Run(XrtGraph *graph, const XrtPassOptions &options) override;

  // For `TopologyVisit`
  const util::Set<ClusterNode *> &Nodes() const { return root_nodes_; }

 private:
  void BuildClusterNodesAndEdges(XrtGraph *graph);
  void ClusteringSubgraphs(const XrtPassOptions &options);
  void RemoveInvalidClusterNodes(const XrtPassOptions &options);

  // Rerank cluster id start by 0
  void RerankClusterIds();
  void DumpClusterInfoToGraph(XrtGraph *graph);

  bool TryToFuseWithParent(ClusterNode *children, ClusterNode *parent,
                           const XrtPassOptions &options);

 private:
  // Root cluster nodes
  util::Set<ClusterNode *> root_nodes_;
  // All allocated nodes and edges which will always alive when
  // running the pass `MarkClusterIdPass`
  std::vector<std::shared_ptr<ClusterNode>> allocated_nodes_;
  std::vector<std::shared_ptr<ClusterEdge>> allocated_edges_;
};

namespace algorithm {
template <>
struct GraphTypeTrait<MarkClusterIdPass> {
  typedef ClusterNode *pNodeType;
  typedef ClusterEdge *pEdgeType;
};
}  // namespace algorithm

void MarkClusterIdPass::BuildClusterNodesAndEdges(XrtGraph *graph) {
  util::Map<int64_t, ClusterNode *> cluster_nodes;
  CHECK(root_nodes_.empty());
  CHECK(allocated_nodes_.empty());
  CHECK(allocated_edges_.empty());
  CHECK(graph) << "Graph is required by running MarkClusterIdPass.";

  algorithm::TopologyVisit(*graph, [&, this](XrtNode *xrt_node) {
    int64_t cluster_id = allocated_nodes_.size();
    auto cluster_node = BuildClusterNode(xrt_node, cluster_id);
    allocated_nodes_.push_back(cluster_node);
    root_nodes_.emplace(cluster_node.get());

    cluster_nodes.emplace(xrt_node->unique_id(), cluster_node.get());
  });

  for (ClusterNode *start : root_nodes_) {
    const XrtNode *xrt_node = start->xrt_node();
    for (const XrtEdge *xrt_edge : xrt_node->out_edges()) {
      int64_t unique_id = xrt_edge->end()->unique_id();
      ClusterNode *end = cluster_nodes[unique_id];
      auto cluster_edge = BuildClusterEdge(start, end);
      SetupClusterEdge(cluster_edge.get(), xrt_edge);

      start->AddOutEdge(cluster_edge.get());
      end->AddInEdge(cluster_edge.get());
      allocated_edges_.push_back(cluster_edge);
    }
  }
}

void MarkClusterIdPass::ClusteringSubgraphs(const XrtPassOptions &options) {
  int32_t max_nodes = options.clustering_options.maximum_nodes;
  int32_t iter_count = 10;

  for (int i = 0; i < iter_count; ++i) {
    bool has_changed = false;
    std::vector<ClusterNode *> ordered_nodes;
    algorithm::TopologyVisit(*this, [&](ClusterNode *node) {
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
            (parent->size() + node->size()) <= max_nodes &&
            TryToFuseWithParent(node, parent, options)) {
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

bool MarkClusterIdPass::TryToFuseWithParent(ClusterNode *children,
                                            ClusterNode *parent,
                                            const XrtPassOptions &options) {
  if (options.clustering_options.strict_clustering) {
    for (const ClusterEdge *edge : parent->out_edges()) {
      if (edge->end() != children &&
          !IsNodeDirectChildren(children, edge->end())) {
        return false;
      }
    }
  }

  bool can_be_fusion = true;
  for (const ClusterEdge *edge : children->in_edges()) {
    if (edge->start() == parent) {
      can_be_fusion = can_be_fusion && !edge->is_fusion_disabled() &&
                      IsSatisfyBackend(edge) && IsSatisfySbpPolicy(edge) &&
                      IsSatisfyTimeShape(edge);
    }
  }
  if (can_be_fusion) {
    return parent->TryMerge(*children);
  }
  return false;
}

void MarkClusterIdPass::RerankClusterIds() {
  int64_t rank = 0;
  for (ClusterNode *node : root_nodes_) {
    node->set_id(rank++);
  }
}

void MarkClusterIdPass::DumpClusterInfoToGraph(XrtGraph *graph) {
  for (const ClusterNode *node : root_nodes_) {
    for (const ClusterNode *folded_node : node->folded_nodes()) {
      int64_t unique_id = folded_node->xrt_node()->unique_id();
      XrtNode *xrt_node = graph->Node(unique_id);
      xrt_node->SetAttr<int64_t>("cluster_id", node->id());
    }
  }
}

void MarkClusterIdPass::RemoveInvalidClusterNodes(
    const XrtPassOptions &options) {
  int32_t min_nodes = options.clustering_options.minimum_nodes;
  int32_t max_nodes = options.clustering_options.maximum_nodes;

  std::vector<ClusterNode *> removing_clusters;
  for (ClusterNode *node : root_nodes_) {
    if (!node->IsCompiled() || node->size() < min_nodes ||
        node->size() > max_nodes) {
      removing_clusters.push_back(node);
    }
  }
  for (ClusterNode *node : removing_clusters) {
    root_nodes_.erase(node);
  }
}

void MarkClusterIdPass::Run(XrtGraph *graph, const XrtPassOptions &options) {
  BuildClusterNodesAndEdges(graph);
  // Clustering nodes iteratively
  ClusteringSubgraphs(options);

  RemoveInvalidClusterNodes(options);
  RerankClusterIds();

  DumpClusterInfoToGraph(graph);
}

REGISTER_XRT_PASS(MarkClusterId, MarkClusterIdPass);

}  // namespace xrt
}  // namespace oneflow
