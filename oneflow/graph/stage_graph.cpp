#include "graph/stage_graph.h"
#include "glog/logging.h"

namespace oneflow {

void OneToOneConnect( 
    const std::vector<StageNode*>& cur_stages,
    const std::vector<StageNode*>& next_stages,
    std::function<void(StageNode*, StageNode*)> ConnectTwoNode) {
  size_t stage_num = cur_stages.size();
  for (size_t i = 0; i < cur_stages.size(); ++i) {
    ConnectTwoNode(cur_stages[i], next_stages[i]);
  }
}

void FullConnect(
    const std::vector<StageNode*>& cur_stages,
    const std::vector<StageNode*>& next_stages,
    std::function<void(StageNode*, StageNode*)> ConnectTwoNode) {
  for (StageNode* cur_stage_node : cur_stages) {
    for (StageNode* next_stage_node : next_stages) {
      ConnectTwoNode(cur_stage_node, next_stage_node);
    }
  }
}

void ConnectRelatedStages(
    const std::vector<StageNode*>& cur_stages,
    const std::vector<StageNode*>& next_stages,
    std::function<void(StageNode*, StageNode*)> ConnectTwoNode) {
  CHECK_EQ(cur_stages.empty(), false);
  CHECK_EQ(next_stages.empty(), false);
  auto cur_parallel_policy = cur_stages.front()->parallel_desc().policy();
  auto next_parallel_policy = next_stages.front()->parallel_desc().policy();
  if (cur_parallel_policy == ParallelDesc::kDataParallel
      && next_parallel_policy == ParallelDesc::kDataParallel) {
    CHECK_EQ(cur_stages.size(), next_stages.size());
    OneToOneConnect(cur_stages, next_stages, ConnectTwoNode);
  } else {
    FullConnect(cur_stages, next_stages, ConnectTwoNode);
  }
}

void StageGraph::Init(std::shared_ptr<const ChainGraph> chain_graph) {
  Graph::Init();
  // Init Stages
  std::unordered_map<const ChainNode*,
                     std::vector<StageNode*>> chain2stages;
  for (const std::unique_ptr<Node>& node : chain_graph->node_vec()) {
    auto chain_node = of_dynamic_cast<const ChainNode*> (node.get());
    chain2stages[chain_node] = {};
    for (MachineId machine_id : chain_node->parallel_desc().machines()) {
      StageNode* stage_node = NewStageNode();
      stage_node->mutable_op_vec_ptr() = chain_node->op_vec_ptr();
      stage_node->mutable_parallel_desc_ptr() = chain_node->parallel_desc_ptr();
      stage_node->mutable_machine_id() = machine_id;
      chain2stages.at(chain_node).push_back(stage_node);
    }
  }
  // Connect Stages
  std::function<void(StageNode*, StageNode*)> ConnectTwoNode = [this]
      (StageNode* src_node, StageNode* dst_node) {
    Connect(src_node, this->NewStageEdge(), dst_node);
  };
  for (const std::unique_ptr<Node>& node : chain_graph->node_vec()) {
    auto cur_chain = of_dynamic_cast<const ChainNode*> (node.get());
    for (const Edge* edge : cur_chain->out_edges()) {
      const std::vector<StageNode*>& cur_stages = chain2stages.at(cur_chain);
      const std::vector<StageNode*>& next_stages =
          chain2stages.at(of_dynamic_cast<const ChainNode*>(edge->dst_node()));
      ConnectRelatedStages(cur_stages, next_stages, ConnectTwoNode);
    }
  }
  // Post processing
  UpdateStartAndStop();
}

} // namespace oneflow
