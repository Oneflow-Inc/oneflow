#include "graph/boxing_transformer_graph.h"

namespace oneflow {

void BoxingTransfmGraph::FwBuildGraph() {
  auto boxing_task_node = of_dynamic_cast<const BoxingTaskNode*>(task_node());
  std::unordered_map<ChainNode*, std::vector<TaskEdge*>> chain2task_edges;
  std::vector<std::pair<ChainNode*, ChainNode*>> chain_pairs;
  // Set chain2tasks, chain_pairs
  for (Edge* base_in_edge : boxing_task_node()->in_edges()) {
    auto in_edge = of_dynamic_cast<TaskEdge*> (base_in_edge);

  }

}

} // namespace oneflow
