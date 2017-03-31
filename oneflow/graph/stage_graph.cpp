#include "graph/stage_graph.h"
#include "glog/logging.h"

namespace oneflow {

StageGraph::StageGraph(std::unique_ptr<const ChainGraph>&& chain_graph) {
  chain_graph_ = std::move(chain_graph);
  // Init Stages
  std::unordered_map<const ChainNode*,
                     std::vector<StageNode*>> chain2stages;
  for (const std::unique_ptr<ChainNode>& cur_chain : chain_graph_->nodes()) {
    chain2stages[cur_chain.get()] = {};
    auto parallel_desc = cur_chain->parallel_desc();
    int32_t range_idx = 0;
    for (MachineId machine_id : parallel_desc->sorted_machines()) {
      StageNode* stage_node = NewFinalNode();
      stage_node->mut_machine_id() = machine_id;
      stage_node->set_chain_node(cur_chain.get());
      stage_node->mut_parallel_range().mut_begin() = range_idx;
      if (parallel_desc->engine() == ParallelDesc::Engine::kDevice) {
        range_idx += parallel_desc->sorted_devices_on_machine(machine_id).size();
      } else {
        CHECK(chain_graph_->IsFirstNode(cur_chain.get()));
        range_idx += 1;
      }
      stage_node->mut_parallel_range().mut_end() = range_idx;
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
