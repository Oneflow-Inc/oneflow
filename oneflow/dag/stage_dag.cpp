#include "dag/stage_dag.h"
#include <set>
#include "common/common.h"
#include "dag/node_meta.h"
#include "dag/dag_iterator.h"
#include "dag/logical_dag.h"
#include "dag/segment_dag.h"
#include "dag/clustering_dag.h"
#include "context/id_map.h"
#include "context/one.h"

namespace oneflow {
template <typename Dtype>
StageDag<Dtype>::StageDag(
  std::shared_ptr<LogicalDag<Dtype>> logical_dag,
  std::shared_ptr<SegmentDag<Dtype>> segment_dag, PathType path_type,
  const std::string& name = "stage_dag")
   : Dag(path_type, name), logical_dag_(logical_dag),
   segment_dag_(segment_dag) {
  Build();
  PrintDag(name_);
}

template <typename Dtype>
void StageDag<Dtype>::Build() {
  CreateStages();
  CollectStageToStageConnection();
  ConnectStageToStage();
  AddStartAndEndNodes();
  PostProcessing();
}

template <typename Dtype>
void StageDag<Dtype>::CreateStages() {
  DagIterator<SegmentDag<Dtype>, true> dag_iterator(*segment_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    // Only process kOpNode
    if (current_node->Type() != NodeType::kOpNode) continue;
    CreateStagesForSegment(current_node->node_name());
  }
}

template <typename Dtype>
void StageDag<Dtype>::CreateStagesForSegment(const std::string& segment_name) {
  auto segment_node = segment_dag_->GetOpNode(segment_name);
  auto segment_meta = segment_node->op();

  auto segment_placement_info = segment_meta->placement_info();
  auto segment_machine_set = segment_placement_info.machine_set();
  // Here, we ensure the stages are added according the ascending order of the
  // machine_id
  for (auto machine_id : segment_machine_set) {
    std::string stage_name = build_stage_name(segment_name, machine_id);
    stage_segment_map_.AddStageSegment(stage_name, segment_name);
    auto op_node = AddOpNode(stage_name, machine_id, segment_name);
  }
}

template <typename Dtype>
OpNode<StageMeta>* StageDag<Dtype>::AddOpNode(
    const std::string& op_name,
    int32_t machine_id,
    const std::string& segment_name) {
  auto op_node = NewOpNode(op_name);
  auto& stage_meta = op_node->mutable_op();
  stage_meta = std::make_shared<StageMeta>();
  stage_meta->mutable_machine_id() = machine_id;
  stage_meta->mutable_segment_name() = segment_name;
  stage_meta->mutable_has_BP()
    = segment_dag_->GetOpNode(segment_name)->op()->has_BP();
  auto it = op_name_to_node_.find(op_name);
  CHECK(it == op_name_to_node_.end())
    << "Duplicate op_name: " << op_name;
  op_name_to_node_.insert({op_name, op_node});
  return op_node;
}

template <typename Dtype>
void StageDag<Dtype>::CollectStageToStageConnection() {
  DagIterator<SegmentDag<Dtype>, true> dag_iterator(*segment_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto segment_node_ptr
      = dynamic_cast<const OpNode<SegmentMeta>*>(current_node);
    CHECK_NOTNULL(segment_node_ptr);
    auto op_meta = segment_node_ptr->op();
    auto op_name = current_node->node_name();
    auto placement_info = op_meta->placement_info();
    auto device_set = placement_info.device_set();
    auto machine_set = placement_info.machine_set();
    auto parallel_policy = placement_info.parallel_policy();

    auto first_stages = stage_segment_map_.StagesFromSegment(op_name);

    auto op_successors = segment_dag_->GetSucceedingOpNodeNames(op_name);
    for (auto& next_op_name : op_successors) {
      auto next_op_node = segment_dag_->GetOpNode(next_op_name);
      if (segment_dag_->IsEnd(next_op_node)) continue; // Skip the end node
      auto next_op_meta = next_op_node->op();
      auto next_placement_info = next_op_meta->placement_info();
      auto next_op_device_set = next_placement_info.device_set();
      auto next_op_machine_set = next_placement_info.machine_set();
      auto next_parallel_policy = next_placement_info.parallel_policy();

      // If first and second are in the same stage, skip
      auto second_stages = stage_segment_map_.StagesFromSegment(next_op_name);

      auto envelopes
        = segment_dag_->FindDataNodesInBetween(op_name, next_op_name);
      CHECK(envelopes.size() == 1);
      auto envelope_name = envelopes[0];

      // NOTE(jiyuan): the following case can be handled by 
      // kDataParallelOnMultipleMachines and kDataParallelOnMultipleDevices
      //if (segment_dag_->IsDataProviderNode(current_node)) {
      //  // DataProvider node must be kDataParallel, no matter what policy the
      //  // the next op has, apply 1x1 connect.
      //  OneToOneConnect(
      //    op_name, first_stages,
      //    next_op_name, second_stages,
      //    envelope_name);
      //  continue;
      //}
      // NOTE(jiyuan): seems that the following case can be handled by 
      // FullConnect
      //if (machine_set.size() == 1 && next_op_machine_set.size() == 1) {
      //  // If both the first machine_set and the next machine_set have only 1 
      //  // machine, no matter what policy they use, and no matter whether the
      //  // machine is the same, apply 1x1 connect.
      //  OneToOneConnect(
      //    op_name, first_stages,
      //    next_op_name, second_stages,
      //    envelope_name);
      //  continue;
      //}
      if ((parallel_policy == kDataParallelOnMultipleDevices
        && next_parallel_policy == kDataParallelOnMultipleDevices)
        || (parallel_policy == kDataParallelOnMultipleMachines
        && next_parallel_policy == kDataParallelOnMultipleDevices)) {
        if (machine_set.size() == next_op_machine_set.size()) {
          OneToOneConnect(
            op_name, first_stages,
            next_op_name, second_stages,
            envelope_name);
        } else {
          // From the data stage to the loss stage, sending label
          // TODO(jiyuan): check the conditions
          LOG(FATAL) << "For continuous data parallel segments, we require "
            << "they have the same number of machine";
          //ContractConnect(
          //  op_name, first_stages,
          //  next_op_name, second_stages,
          //  envelope_name);
        }
        continue;
      }
      // otherwise, FullConnect, covers the following cases
      // (1) 1x1 machine, no matter what parallel policy it is
      // (2) between kDataParallelOnMultipleDevices and 
      //     kModelParallelOnMultipleDevices
      // (3) kNaiveParallelOnSingleMachine and kDataParallelOnMultipleDevices
      // (4) kNaiveParallelOnSingleMachine and kModelParallelOnMultipleDevices
      FullConnect(
        op_name, first_stages,
        next_op_name, second_stages,
        envelope_name);
    }
  }
}

template <typename Dtype>
void StageDag<Dtype>::AddStageSegmentPair(
  const std::string& stage_name,
  const std::string& segment_name,
  const std::string& next_stage_name,
  const std::string& next_segment_name) {
  StageSegmentPair from{ stage_name, segment_name };
  StageSegmentPair to{ next_stage_name, next_segment_name };
  auto it = stage_segment_to_successors_.find(from);
  if (it == stage_segment_to_successors_.end()) {
    std::vector<StageSegmentPair> successors{ to };
    stage_segment_to_successors_.insert({ from, successors });
  } else {
    it->second.push_back(to);
  }
}

template <typename Dtype>
void StageDag<Dtype>::UpdateStageToStage(
  const std::string& stage_from,
  const std::string& stage_to,
  const std::string& envelope_name) {
  auto from_it = stage_to_stage_.find(stage_from);
  if (from_it == stage_to_stage_.end()) {
    std::unordered_map<std::string, std::string> to_stage_data_pair;
    to_stage_data_pair.insert({ stage_to, envelope_name });
    stage_to_stage_.insert({ stage_from, to_stage_data_pair });
  } else {
    auto& to_stage_data_pair = from_it->second;
    auto to_it = to_stage_data_pair.find(stage_to);
    CHECK(to_it == to_stage_data_pair.end());
    to_stage_data_pair.insert({ stage_to, envelope_name });
  }
}

template <typename Dtype>
void StageDag<Dtype>::ConnectStageToStage() {
  for (auto& stage_to_stage_pair : stage_to_stage_) {
    auto stage_from = stage_to_stage_pair.first;
    auto to_stages = stage_to_stage_pair.second;
    auto node_from = GetOpNode(stage_from);
    auto machine_id_from = node_from->op()->machine_id();
    for (auto& to_stage_data_pair : to_stages) {
      auto stage_to = to_stage_data_pair.first;
      auto to_stage_data = to_stage_data_pair.second;
      auto node_to = GetOpNode(stage_to);
      auto machine_id_to = node_to->op()->machine_id();

      auto blob_names
        = segment_dag_->GetDataNode(to_stage_data)->data()->blob_names();
      std::string envelope_name
        = build_envelope_name(to_stage_data, machine_id_from, machine_id_to);

      auto envelope_node = AddDataNode(envelope_name, blob_names);
      std::vector<ONode*> inputs{ node_from };
      std::vector<ONode*> outputs{ node_to };
      AddEdges(envelope_node, inputs, outputs);
    }
  }
  // Add the data nodes of the blobs which directly connect to END node
  auto envelope_names_without_successor
    = segment_dag_->GetPreceedingDataNodeNamesOfEndNode();
  for (auto& envelope_name : envelope_names_without_successor) {
    auto data_node_predecessors
      = segment_dag_->GetPreceedingOpNodeNamesOfDataNode(envelope_name);
    CHECK(data_node_predecessors.size() == 1);
    // from segment to stage
    auto stages
      = stage_segment_map_.StagesFromSegment(data_node_predecessors[0]);
    for (auto stage : stages) {
      auto stage_node = GetOpNode(stage);
      auto machine_id = stage_node->op()->machine_id();
      auto stage_envelope_name
        = envelope_name + "_" + std::to_string(machine_id) + "x";
      auto envelope_node = segment_dag_->GetDataNode(envelope_name);
      auto blob_names = envelope_node->data()->blob_names();
      auto data_node = AddDataNode(stage_envelope_name, blob_names);
      data_node->AddParent(stage_node);
    }
  }
}

template <typename Dtype>
void StageDag<Dtype>::OneToOneConnect(
  const std::string& first_segment_name,
  const std::vector<std::string>& first_stage_names,
  const std::string& second_segment_name,
  const std::vector<std::string>& second_stage_names,
  const std::string& envelope_name) {
  CHECK(first_stage_names.size() == second_stage_names.size());
  int32_t parallel_size = first_stage_names.size();
  for (int32_t id = 0; id < parallel_size; ++id) {
    auto first_it = op_name_to_node_.find(first_stage_names[id]);
    CHECK(first_it != op_name_to_node_.end());
    auto first_node = first_it->second;
    auto first_meta = first_node->op();
    auto first_machine_id = first_meta->machine_id();

    auto second_it = op_name_to_node_.find(second_stage_names[id]);
    CHECK(second_it != op_name_to_node_.end());
    auto second_node = second_it->second;
    auto second_meta = second_node->op();
    auto second_machine_id = second_meta->machine_id();

    UpdateStageToStage(first_stage_names[id],
      second_stage_names[id], envelope_name);

    AddStageSegmentPair(
      first_stage_names[id], first_segment_name,
      second_stage_names[id], second_segment_name);
  }
}
//template <typename Dtype>
//void StageDag<Dtype>::ContractConnect(
//  const std::string& first_segment_name,
//  const std::vector<std::string>& first_stage_names,
//  const std::string& second_segment_name,
//  const std::vector<std::string>& second_stage_names,
//  const std::string& envelope_name) {
//  int32_t first_size = first_stage_names.size();
//  int32_t second_size = second_stage_names.size();
//  CHECK(first_size > second_size);
//  CHECK(first_size % second_size == 0);
//  for (int32_t first_id = 0; first_id < first_size; ++first_id) {
//    auto first_it = op_name_to_node_.find(first_stage_names[first_id]);
//    CHECK(first_it != op_name_to_node_.end());
//    auto first_node = first_it->second;
//    auto first_meta = first_node->op();
//    auto first_machine_id = first_meta->machine_id();
//
//    for (int32_t second_id = 0; second_id < second_size; ++second_id) {
//      if (first_id % second_size == second_id) {
//        auto second_it = op_name_to_node_.find(second_stage_names[second_id]);
//        CHECK(second_it != op_name_to_node_.end());
//        auto second_node = second_it->second;
//        auto second_meta = second_node->op();
//        auto second_machine_id = second_meta->machine_id();
//
//        UpdateStageToStage(first_stage_names[first_id],
//          second_stage_names[second_id], envelope_name);
//        AddStageSegmentPair(
//          first_stage_names[first_id], first_segment_name,
//          second_stage_names[second_id], second_segment_name);
//      }
//    }
//  }
//}

template <typename Dtype>
void StageDag<Dtype>::FullConnect(
  const std::string& first_segment_name,
  const std::vector<std::string>& first_stage_names,
  const std::string& second_segment_name,
  const std::vector<std::string>& second_stage_names,
  const std::string& envelope_name) {
  for (auto& first_name : first_stage_names) {
    auto first_it = op_name_to_node_.find(first_name);
    CHECK(first_it != op_name_to_node_.end());
    auto first_node = first_it->second;
    auto first_meta = first_node->op();
    auto first_machine_id = first_meta->machine_id();
    for (auto& second_name : second_stage_names) {
      auto second_it = op_name_to_node_.find(second_name);
      CHECK(second_it != op_name_to_node_.end());
      auto second_node = second_it->second;
      auto second_meta = second_node->op();
      auto second_machine_id = second_meta->machine_id();

      UpdateStageToStage(first_name,
        second_name, envelope_name);
      AddStageSegmentPair(
        first_name, first_segment_name,
        second_name, second_segment_name);
    }
  }
}

template <typename Dtype>
DataNode<EnvelopeMeta>* StageDag<Dtype>::AddDataNode(
    const std::string& data_name,
    const std::vector<std::string>& blob_names) {
  auto data_node = NewDataNode(data_name);
  auto& envelope_meta = data_node->mutable_data();
  envelope_meta = std::make_shared<EnvelopeMeta>();
  envelope_meta->mutable_blob_names() = blob_names;
  auto it = data_name_to_node_.find(data_name);
  CHECK(it == data_name_to_node_.end())
    << "Duplicate data_name: " << data_name;
  data_name_to_node_.insert({data_name, data_node});
  return data_node;
}

template <typename Dtype>
std::string StageDag<Dtype>::build_stage_name(
  const std::string& segment_name, int32_t machine_id) const {
  std::string stage_name = segment_name + "_m" + std::to_string(machine_id);
  return stage_name;
}

template <typename Dtype>
std::string StageDag<Dtype>::build_envelope_name(
  const std::string& segment_envelope_name, int32_t from_machine_id,
  int32_t to_machine_id) const {
  std::string envelope_name = segment_envelope_name;
  envelope_name +=
    "_" + std::to_string(from_machine_id) + "x" + std::to_string(to_machine_id);
  return envelope_name;
}

template <typename Dtype>
int32_t StageDag<Dtype>::machine_id(const std::string& stage_name) const {
  auto stage_node = GetOpNode(stage_name);
  auto machine_id = stage_node->op()->machine_id();
  return machine_id;
}

template <typename Dtype>
std::string StageDag<Dtype>::GetSegmentNameFromStageName(
  const std::string& stage_name) const {
  return stage_segment_map_.SegmentFromStage(stage_name);
}

template <typename Dtype>
std::vector<std::string> StageDag<Dtype>::GetStageNamesFromSegmentName(
  const std::string& segment_name) const {
  return stage_segment_map_.StagesFromSegment(segment_name);
}

template <typename Dtype>
void StageDag<Dtype>::StageSegmentMap::AddStageSegment(const std::string& stage,
  const std::string& segment) {
  CHECK(stage_to_segment_.count(stage) == 0);
  stage_to_segment_.insert({ stage, segment });
  auto stage_it = segment_to_stages_.find(segment);
  if (stage_it != segment_to_stages_.end()) {
    stage_it->second.push_back(stage);
  } else {
    std::vector<std::string> stages{ stage };
    segment_to_stages_.insert({ segment, stages });
  }
}

template <typename Dtype>
std::string StageDag<Dtype>::StageSegmentMap::SegmentFromStage(
  const std::string& stage) const {
  auto segment_it = stage_to_segment_.find(stage);
  CHECK(segment_it != stage_to_segment_.end());
  return segment_it->second;
}

template <typename Dtype>
std::vector<std::string> StageDag<Dtype>::StageSegmentMap::StagesFromSegment(
  const std::string& segment) const {
  auto stages_it = segment_to_stages_.find(segment);
  CHECK(stages_it != segment_to_stages_.end());
  return stages_it->second;
}

INSTANTIATE_CLASS(StageDag);
}  // namespace oneflow
