#include "graph/stage_graph.h"
#include "glog/logging.h"

namespace oneflow {

void StageGraph::Init(std::unique_ptr<const ChainGraph>&& chain_graph) {
  chain_graph_ = std::move(chain_graph);
  // Init Stages
  std::unordered_map<const ChainNode*,
                     std::vector<StageNode*>> chain2stages;
  for (const std::unique_ptr<ChainNode>& cur_chain : chain_graph_->nodes()) {
    chain2stages[cur_chain.get()] = {};
    for (MachineId machine_id : cur_chain->parallel_desc()->machines()) {
      StageNode* stage_node = NewFinalNode();
      stage_node->mut_machine_id() = machine_id;
      stage_node->set_chain_node(cur_chain.get());
      chain2stages.at(cur_chain.get()).push_back(stage_node);
    }
  }
  for (const std::unique_ptr<ChainNode>& cur_chain : chain_graph_->nodes()) {
    for (const ChainEdge* edge : cur_chain->out_edges()) {
      const std::vector<StageNode*>& cur_stages =
          chain2stages.at(cur_chain.get());
      const std::vector<StageNode*>& succ_stages =
          chain2stages.at(edge->dst_node());
      for (StageNode* cur_stage_node : cur_stages) {
        for (StageNode* succ_stage_node : succ_stages) {
          Connect(cur_stage_node, this->NewFinalEdge(), succ_stage_node);
        }
      }
    }
  }
  // Post processing
  UpdateSourceAndSink();
}

} // namespace oneflow
