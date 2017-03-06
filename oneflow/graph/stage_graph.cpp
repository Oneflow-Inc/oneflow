#include "graph/stage_graph.h"
#include "glog/logging.h"

namespace oneflow {

void OneToOneConnect( 
    const std::vector<StageNode*>& cur_stages,
    const std::vector<StageNode*>& next_stages) {
  size_t stage_num = cur_stages.size();
  for (size_t i = 0; i < cur_stages.size(); ++i) {
    ConnectTwoNode(cur_stages[i], next_stages[i]);
  }
}

void FullConnect(
    const std::vector<StageNode*>& cur_stages,
    const std::vector<StageNode*>& next_stages) {
  for (StageNode* cur_stage_node : cur_stages) {
    for (StageNode* next_stage_node : next_stages) {
      ConnectTwoNode(cur_stage_node, next_stage_node);
    }
  }
}

void ConnectRelatedStages(
    const std::vector<StageNode*>& cur_stages,
    const std::vector<StageNode*>& next_stages) {
  CHECK_EQ(cur_stages.empty(), false);
  CHECK_EQ(next_stages.empty(), false);
  auto cur_parallel_policy = cur_stages.front()->parallel_desc().policy();
  auto next_parallel_policy = next_stages.front()->parallel_desc().policy();
  if (cur_parallel_policy == ParallelDesc::kDataParallel
      && next_parallel_policy == ParallelDesc::kDataParallel) {
    CHECK_EQ(cur_stages.size(), next_stages.size());
    OneToOneConnect(cur_stages, next_stages);
  } else {
    FullConnect(cur_stages, next_stages);
  }
}

void StageGraph::Init(const std::string& dag_name,
                    std::shared_ptr<const SegmentGraph> segment_dag) {
  // Init Stages
  std::unordered_map<const SegmentNode*,
                     std::vector<StageNode*>> seg2stages;
  for (const std::unique_ptr<Node>& node : segment_dag->node_vec()) {
    auto seg_node = of_dynamic_cast<const SegmentNode*> (node.get());
    seg2stages[seg_node] = {};
    for (MachineId machine_id : seg_node->parallel_desc().machines()) {
      StageNode* stage_node = NewStageNode();
      stage_node->mutable_op_vec() = seg_node->op_vec();
      stage_node->mutable_parallel_desc_ptr() = seg_node->parallel_desc_ptr();
      stage_node->mutable_machine_id() = machine_id;
      seg2stages.at(seg_node).push_back(stage_node);
    }
  }
  // Connect Stages
  for (const std::unique_ptr<Node>& node : segment_dag->node_vec()) {
    auto cur_seg = of_dynamic_cast<const SegmentNode*> (node.get());
    for (const Node* next_seg : cur_seg->successors()) {
      const std::vector<StageNode*>& cur_stages = seg2stages.at(cur_seg);
      const std::vector<StageNode*>& next_stages =
          seg2stages.at(of_dynamic_cast<const SegmentNode*>(next_seg));
      ConnectRelatedStages(cur_stages, next_stages);
    }
  }
  // Post processing
  ConnectStartAndStop();
}

} // namespace oneflow
