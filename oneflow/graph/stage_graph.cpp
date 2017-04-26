#include "graph/stage_graph.h"
#include "glog/logging.h"

namespace oneflow {

StageGraph::StageGraph(std::unique_ptr<const ChainGraph>&& chain_gph) {
  LOG(INFO) << "Build StageGraph...";
  chain_gph_ = std::move(chain_gph);
  HashMap<const ChainNode*, std::vector<StageNode*>> chain2stages;
  // Construct Stage
  for (const std::unique_ptr<ChainNode>& cur_chain : chain_gph_->nodes()) {
    chain2stages[cur_chain.get()] = {};
    auto parallel_desc = cur_chain->parallel_desc();
    uint64_t range_idx = 0;
    for (uint64_t machine_id : parallel_desc->sorted_machine_ids()) {
      StageNode* stage_node = NewNode();
      stage_node->mut_machine_id() = machine_id;
      stage_node->set_chain_node(cur_chain.get());
      stage_node->mut_parallel_range().mut_begin() = range_idx;
      size_t device_num =
          parallel_desc->sorted_device_phy_ids(machine_id).size();
      if (device_num == 0) {
        device_num = 1; // DiskLoader
      }
      range_idx += parallel_desc->sorted_device_phy_ids(machine_id).size();
      stage_node->mut_parallel_range().mut_end() = range_idx;
      chain2stages.at(cur_chain.get()).push_back(stage_node);
    }
  }
  // Connect Stage
  for (const std::unique_ptr<ChainNode>& cur_chain : chain_gph_->nodes()) {
    for (const ChainEdge* edge : cur_chain->out_edges()) {
      const auto& cur_stages = chain2stages.at(cur_chain.get());
      const auto& succ_stages = chain2stages.at(edge->dst_node());
      for (StageNode* cur_stage_node : cur_stages) {
        for (StageNode* succ_stage_node : succ_stages) {
          Connect(cur_stage_node, NewEdge(), succ_stage_node);
        }
      }
    }
  }
  // Post processing
  UpdateSourceAndSink();
  ToDotFile(LogDir() + "/stage_graph.dot");
}

} // namespace oneflow
