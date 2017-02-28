#include "dag/stage_dag.h"
#include "glog/logging.h"

namespace oneflow {

void OneToOneConnect( 
    const std::vector<StageOpNode*>& cur_stages,
    const std::vector<StageOpNode*>& next_stages,
    std::function<void(StageOpNode*, StageOpNode*)> connect_stage) {
  size_t stage_num = cur_stages.size();
  for (size_t i = 0; i < cur_stages.size(); ++i) {
    connect_stage(cur_stages[i], next_stages[i]);
  }
}

void FullConnect(
    const std::vector<StageOpNode*>& cur_stages,
    const std::vector<StageOpNode*>& next_stages,
    std::function<void(StageOpNode*, StageOpNode*)> connect_stage) {
  for (StageOpNode* cur_stage_node : cur_stages) {
    for (StageOpNode* next_stage_node : next_stages) {
      connect_stage(cur_stage_node, next_stage_node);
    }
  }
}

void ConnectRelatedStages(
    const std::vector<StageOpNode*>& cur_stages,
    const std::vector<StageOpNode*>& next_stages,
    std::function<void(StageOpNode*, StageOpNode*)> connect_stage) {
  CHECK_EQ(cur_stages.empty(), false);
  CHECK_EQ(next_stages.empty(), false);
  auto cur_parallel_policy = cur_stages.front()->parallel_desc().policy();
  auto next_parallel_policy = next_stages.front()->parallel_desc().policy();
  if (cur_parallel_policy == ParallelDesc::kDataParallel
      && next_parallel_policy == ParallelDesc::kDataParallel) {
    CHECK_EQ(cur_stages.size(), next_stages.size());
    OneToOneConnect(cur_stages, next_stages, connect_stage);
  } else {
    FullConnect(cur_stages, next_stages, connect_stage);
  }
}

void StageDag::Init(const std::string& dag_name,
                    std::shared_ptr<const SegmentDag> segment_dag) {
  // Init Stages
  std::unordered_map<const SegmentOpNode*,
                     std::vector<StageOpNode*>> seg2stages;
  for (const std::unique_ptr<OpNode>& opnode : segment_dag->op_node_vec()) {
    auto seg_opnode = of_dynamic_cast<const SegmentOpNode*> (opnode.get());
    seg2stages[seg_opnode] = {};
    for (MachineId machine_id : seg_opnode->parallel_desc().machine_set()) {
      StageOpNode* stage_opnode = NewStageOpNode();
      stage_opnode->mutable_layer_desc_vec() = seg_opnode->layer_desc_vec();
      stage_opnode->mutable_parallel_desc() = seg_opnode->parallel_desc();
      stage_opnode->mutable_machine_id() = machine_id;
      seg2stages.at(seg_opnode).push_back(stage_opnode);
    }
  }
  // Connect Stages
  std::function<void(StageOpNode*, StageOpNode*)> connect_stage = [this](
      StageOpNode* pre, StageOpNode* next) {
    StageDataNode* node = this->NewStageDataNode();
    node->AddPredecessor(pre);
    next->AddPredecessor(node);
  };
  for (const std::unique_ptr<OpNode>& opnode : segment_dag->op_node_vec()) {
    auto cur_seg_op = of_dynamic_cast<const SegmentOpNode*> (opnode.get());
    for (const SegmentOpNode* next_seg_op : cur_seg_op->op_successors()) {
      const std::vector<StageOpNode*>& cur_stages = seg2stages.at(cur_seg_op);
      const std::vector<StageOpNode*>& next_stages = seg2stages.at(next_seg_op);
      ConnectRelatedStages(cur_stages, next_stages, connect_stage);
    }
  }
  // Post processing
  ConnectStartAndStop();
}

} // namespace oneflow
