/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/passes/cluster.h"
#include "oneflow/xrt/passes/pass.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {

class MarkClusterIdPass : public XrtPass {
 public:
  MarkClusterIdPass() = default;

  void Run(XrtGraph *graph, const XrtPassOptions &options) override;

  // For `TopologyVisit`.
  const util::Set<ClusterNode *> &Nodes() const { return root_nodes_; }

 private:
  void BuildClusterNodesAndEdges(XrtGraph *graph);
  void ClusteringSubgraphs(const ClusteringOptions &options, const XrtEngine &engine);

  void RemoveInvalidClusterNodes(const ClusteringOptions &options);

  void FinalizeClusterEngine(const ClusteringOptions &options, const XrtEngine &engine);

  // Rerank cluster id start by 0.
  void RerankClusterIds();
  void DumpClusterInfoToGraph(XrtGraph *graph);

  bool TryToFuseWithParent(ClusterNode *children, ClusterNode *parent,
                           const ClusteringOptions &options);

 private:
  // Root cluster nodes.
  util::Set<ClusterNode *> root_nodes_;
  // All allocated nodes and edges which will always alive when
  // running the pass `MarkClusterIdPass`.
  std::vector<ClusterNodePtr> allocated_nodes_;
  std::vector<ClusterEdgePtr> allocated_edges_;
};

namespace algorithm {
template<>
struct GraphTypeTrait<MarkClusterIdPass> {
  typedef ClusterNode *pNodeType;
  typedef ClusterEdge *pEdgeType;
};
}  // namespace algorithm

void MarkClusterIdPass::BuildClusterNodesAndEdges(XrtGraph *graph) {
  CHECK(graph) << "Graph is required by MarkClusterIdPass.";
  util::Map<int64_t, ClusterNode *> cluster_nodes;
  algorithm::TopologyVisit(*graph, [&](XrtNode *node) {
    int64_t cluster_id = allocated_nodes_.size();
    auto cluster_node = BuildClusterNode(node, cluster_id);
    root_nodes_.insert(cluster_node.get());
    cluster_nodes[node->unique_id()] = cluster_node.get();
    allocated_nodes_.push_back(std::move(cluster_node));
  });

  for (ClusterNode *start : root_nodes_) {
    for (const XrtEdge *edge : start->xrt_node()->out_edges()) {
      int64_t unique_id = edge->end()->unique_id();
      ClusterNode *end = cluster_nodes.at(unique_id);

      auto cluster_edge = BuildClusterEdge(start, end);
      SetupClusterEdge(cluster_edge.get(), edge);

      start->AddOutEdge(cluster_edge.get());
      end->AddInEdge(cluster_edge.get());
      allocated_edges_.push_back(std::move(cluster_edge));
    }
  }
}

void MarkClusterIdPass::ClusteringSubgraphs(const ClusteringOptions &options,
                                            const XrtEngine &engine) {
  if (!CheckUseXrtEngine(options, engine)) { return; }
  for (int i = 0; i < options.max_iteration; ++i) {
    bool has_changed = false;
    std::vector<ClusterNode *> ordered_nodes;
    algorithm::TopologyVisit(*this, [&](ClusterNode *node) {
      if (!node->IsCompiled(engine, options.train_phase)
          || node->IsOptimizer(engine) /* skip model update op */) {
        return;
      }
      ordered_nodes.push_back(node);
    });

    // for (ClusterNode *node : ordered_nodes) {
    for (int i = ordered_nodes.size() - 1; i >= 0; --i) {
      ClusterNode *node = ordered_nodes[i];
      util::Set<ClusterNode *> candidate_parents;
      for (ClusterEdge *edge : node->in_edges()) { candidate_parents.insert(edge->start()); }
      for (ClusterNode *parent : candidate_parents) {
        if (parent->IsCompiled(engine, options.train_phase)
            && (parent->size() + node->size()) <= options.maximum_nodes
            && TryToFuseWithParent(node, parent, options)) {
          has_changed = true;
          root_nodes_.erase(node);
          break;
          // node = parent;
        }
      }
    }
    if (!has_changed) { break; }
  }

  FinalizeClusterEngine(options, engine);
}

bool MarkClusterIdPass::TryToFuseWithParent(ClusterNode *children, ClusterNode *parent,
                                            const ClusteringOptions &options) {
  if (options.strict_clustering) {
    // for (const ClusterEdge *edge : children->in_edges()) {
    //   if (edge->start() != parent && !edge->start()->IsReachable(*parent)) {
    //     return false;
    //   }
    // }
    for (const ClusterEdge *edge : parent->out_edges()) {
      if (edge->end() != children && /* !children->IsReachable(*(edge->end())) */
          !IsNodeDirectChildren(children, edge->end())) {
        return false;
      }
    }
  }

  bool can_be_fusion = true;
  for (const ClusterEdge *edge : children->in_edges()) {
    if (edge->start() == parent) {
      can_be_fusion = can_be_fusion && !edge->is_fusion_disabled() && IsSatisfyBackend(edge)
                      && IsSatisfySbpPolicy(edge) && IsSatisfyTimeShape(edge);
    }
  }
  if (can_be_fusion) { return parent->TryMerge(*children); }
  return false;
}

void MarkClusterIdPass::RerankClusterIds() {
  int64_t rank = 0;
  for (ClusterNode *node : root_nodes_) { node->set_cluster_id(rank++); }
}

void MarkClusterIdPass::DumpClusterInfoToGraph(XrtGraph *graph) {
  for (const ClusterNode *node : root_nodes_) {
    for (const ClusterNode *folded_node : node->folded_nodes()) {
      int64_t unique_id = folded_node->xrt_node()->unique_id();
      XrtNode *xrt_node = graph->Node(unique_id);
      xrt_node->Attr<XrtEngine>("engine", node->engine());
      xrt_node->Attr<int64_t>("cluster_id", node->cluster_id());
    }
  }
}

void MarkClusterIdPass::FinalizeClusterEngine(const ClusteringOptions &options,
                                              const XrtEngine &engine) {
  const int min_nodes = options.minimum_nodes;
  const int max_nodes = options.maximum_nodes;
  for (ClusterNode *node : root_nodes_) {
    if (node->IsCompiled(engine, options.train_phase) && node->size() >= min_nodes
        && node->size() <= max_nodes) {
      node->set_engine(engine);
    }
  }
}

void MarkClusterIdPass::RemoveInvalidClusterNodes(const ClusteringOptions &options) {
  const int min_nodes = options.minimum_nodes;
  const int max_nodes = options.maximum_nodes;
  std::vector<ClusterNode *> removing_clusters;
  for (ClusterNode *node : root_nodes_) {
    if (node->engine() == XrtEngine::DEFAULT || node->size() < min_nodes
        || node->size() > max_nodes) {
      removing_clusters.push_back(node);
    }
  }
  for (ClusterNode *node : removing_clusters) { root_nodes_.erase(node); }
}

void MarkClusterIdPass::Run(XrtGraph *graph, const XrtPassOptions &options) {
  BuildClusterNodesAndEdges(graph);
  // Clustering nodes iteratively.
  const auto &clustering_options = options.clustering_options;
  if (clustering_options.train_phase) {
    ClusteringSubgraphs(clustering_options, XrtEngine::XLA);
    ClusteringSubgraphs(clustering_options, XrtEngine::TENSORRT);
  } else {
    ClusteringSubgraphs(clustering_options, XrtEngine::TENSORRT);
    ClusteringSubgraphs(clustering_options, XrtEngine::XLA);
  }

  RemoveInvalidClusterNodes(clustering_options);
  RerankClusterIds();

  DumpClusterInfoToGraph(graph);
}

REGISTER_XRT_PASS(MarkClusterId, MarkClusterIdPass);

}  // namespace xrt
}  // namespace oneflow
