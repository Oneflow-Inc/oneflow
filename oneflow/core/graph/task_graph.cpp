#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

TaskGraph::TaskGraph(std::unique_ptr<const ChainGraph>&& chain_gph) {
  chain_gph_ = std::move(chain_gph);
  HashMap<const ChainNode*, std::vector<CompTaskNode*>> chain2comp_tasks;
  chain_gph_->ForEachNode([&](const ChainNode* chain_node) {
    chain_node->GenSortedCompTaskNodes([&](CompTaskNode* comp_task_node) {
      AddAllocatedNode(comp_task_node);
      chain2comp_tasks[chain_node].push_back(comp_task_node);
    });
  });
}

}  // namespace oneflow
