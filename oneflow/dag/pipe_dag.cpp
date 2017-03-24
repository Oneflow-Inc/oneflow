#include "dag/pipe_dag.h"
#include <set>
#include <unordered_set>
#include "common/common.h"
#include "common/str_util.h"
#include "dag/segment_dag.h"
#include "dag/node_meta.h"
#include "dag/dag_iterator.h"
#include "dag/stage_dag.h"
#include "context/one.h"
#include "context/id_map.h"

namespace oneflow {
template <typename Dtype>
PipeDag<Dtype>::PipeDag(
    std::shared_ptr<SegmentDag<Dtype>> segment_dag,
    std::shared_ptr<StageDag<Dtype>> stage_dag,
    PathType path_type,
    const std::string& name) :
  Dag(path_type, name),
  segment_dag_(segment_dag),
  stage_dag_(stage_dag) {
  Build();
  PrintDag(name_, true);
}

template <typename Dtype>
void PipeDag<Dtype>::Build() {
  // Each stage is expanded to one (e.g., data provider stage) or several pipes
  // (e.g., general compute stage).
  ExpandStageToPipes();
  // For each compute pipe, add an in_copy or out_copy pipe if necessary.
  AddCopyPipeNodes();
  // For each stage, add an in_net for each preceding stage if the two stages
  // are not at the same machine; add an out_net for each succeeding stage if
  // the two stages are not at the same machine.
  AddNetPipeNodes();
  // Connect out_net and in_net if they are supposed to be connected.
  ConnectNetPipeNodes();
  // Add in_boxing and out_boxing pipe for each stage if necessary. Connect the
  // in_boxing pipe and the pipes in the stage. Delegate the inputs of the pipes
  // to the added in_boxing node. Connect the out_boxing pipe and the pipes in
  // the stage. Delegate the outputs of the pipes to the added out_boxing node.
  AddBoxingPipeNodes();
  AddBoxingInfos();
  // From the outgoing direction, connect the compute pipe node and out_net node
  // if possible. If there is a boxing node between the compute pipe node and
  // out_net node, just connect the boxing node and out_net node
  ConnectOutgoingPipeNodes();
  // From the incoming direction, connect the in_net node and the compute pipe
  // node if possible. If there is a boxing node inbetween the compute pipe and
  // the in_net node, just connect the in_net node and in_boxing node
  ConnectIncomingPipeNodes();

  AddDataNodesWithoutSuccessors();

  AddStartAndEndNodes();
  // Do post-processing to allow the inverse topological traverse on PipeDag
  PostProcessing();
}

template <typename Dtype>
void PipeDag<Dtype>::ExpandStageToPipes() {
  // Each stage in StageDag is expanded to either one or several pipe nodes
  DagIterator<StageDag<Dtype>, true> dag_iterator(*stage_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto stage_node_ptr
      = dynamic_cast<const OpNode<StageMeta>*> (current_node);
    auto stage_name = current_node->node_name();
    auto stage_machine_id = stage_node_ptr->op()->machine_id();
    auto segment_name = stage_node_ptr->op()->segment_name();
    auto segment_node = segment_dag_->GetOpNode(segment_name);
    auto segment_meta = segment_node->op();
    auto& placement_info = segment_meta->placement_info();
    auto parallel_policy = placement_info.parallel_policy();
    if (parallel_policy == kNaiveParallelOnSingleMachine
      || parallel_policy == kDataParallelOnMultipleMachines
      || parallel_policy == kModelParallelOnMultipleMachines) {
      // The task is on host, it will be treated as kDataTask
      SingleStageToSinglePipe(stage_name, segment_name, stage_machine_id);
    } else {
      // The task in on device and will be treated as kComputeTask
      SingleStageToMultiplePipes(stage_name, segment_name, stage_machine_id);
    }
  }
}

template <typename Dtype>
void PipeDag<Dtype>::SingleStageToSinglePipe(
  const std::string& stage_name,
  const std::string& segment_name,
  int32_t machine_id) {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  auto local_id = id_map->data_thread_local_id();
  auto pipe_name = build_pipe_name_with_local_id(
    compute_pipe_prefix_, stage_name, machine_id, local_id);
  auto thread_id
    = id_map->thread_id_from_machine_and_local(machine_id, local_id);
  auto pipe_node = AddOpNode(pipe_name, thread_id, TaskType::kDataTask);
  pipe_node->op()->mutable_has_BP()
    = stage_dag_->GetOpNode(stage_name)->op()->has_BP();
  CHECK(pipe_node->op()->has_BP() == false);

  stage_pipe_map_.AddStagePipe(stage_name, pipe_name);
}

template <typename Dtype>
void PipeDag<Dtype>::SingleStageToMultiplePipes(
  const std::string& stage_name,
  const std::string& segment_name,
  int32_t machine_id) {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  auto device_num_each_machine = id_map->device_num_each_machine();
  // Here, we ensure the pipes are added according to the ascending order of
  // thread_id
  for (int32_t local_id = 0; local_id < device_num_each_machine; ++local_id) {
    auto pipe_name = build_pipe_name_with_local_id(
      compute_pipe_prefix_, stage_name, machine_id, local_id);
    auto thread_id
      = id_map->thread_id_from_machine_and_local(machine_id, local_id);
    auto op_node = AddOpNode(pipe_name, thread_id, TaskType::kComputeTask);
    op_node->op()->mutable_has_BP()
      = stage_dag_->GetOpNode(stage_name)->op()->has_BP();

    stage_pipe_map_.AddStagePipe(stage_name, pipe_name);
  }
}

template <typename Dtype>
void PipeDag<Dtype>::AddCopyPipeNodes() {
  for (auto& name_node_pair : op_name_to_node_) {
    auto pipe_name = name_node_pair.first;
    auto pipe_node = name_node_pair.second;
    // For each compute pipe, add in_copy and out_copy if necessary
    if (IsComputePipe(pipe_name)) {
      AddCopyPipeNodeForComputePipe(pipe_name);
    }
  }
}

template <typename Dtype>
void PipeDag<Dtype>::AddCopyPipeNodeForComputePipe(
  const std::string& pipe_name) {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  auto stage_name = GetStageName(pipe_name);
  auto machine_id = stage_dag_->machine_id(stage_name);
  // For gradient generator pipe, only need out_copy
  bool is_first = stage_dag_->IsFirstOpNode(stage_name);
  // For Loss pipe, only need in_copy
  bool is_last = stage_dag_->IsLastOpNode(stage_name);
  auto pipe_node = GetOpNode(pipe_name);
  auto thread_id = pipe_node->op()->thread_id();
  auto local_id = id_map->local_id_from_thread_id(thread_id);

  if (!is_first) {
    auto in_copy_name = in_copy_pipe_prefix_ + pipe_name;
    auto in_copy_node
      = AddOpNode(in_copy_name, thread_id, TaskType::kCopyTask);
    auto stage_name = GetStageName(pipe_name);
    auto machine_id = stage_dag_->machine_id(stage_name);
    auto segment_name = stage_dag_->GetSegmentNameFromStageName(stage_name);
    auto preceding_envelopes
      = segment_dag_->GetPrecedingDataNodeNames(segment_name);
    CHECK(preceding_envelopes.size() >= 1);
    auto in_copy_data_name = build_envelope_name_from_envelopes(
      in_copy_envelope_prefix_, preceding_envelopes, machine_id, local_id);
    auto in_copy_data_node = AddDataNode(in_copy_data_name);
    // Connect in_copy actor <-> compute pipe, having a data node inbetween.
    std::vector<ONode*> inputs{ in_copy_node };
    std::vector<ONode*> outputs{ pipe_node };
    AddEdges(in_copy_data_node, inputs, outputs);

    // Track the mapping between in_copy and compute pipe
    copy_compute_map_.AddInCopy(in_copy_name, pipe_name);
  }
  if (!is_last) {
    auto out_copy_name = out_copy_pipe_prefix_ + pipe_name;
    auto out_copy_node = AddOpNode(out_copy_name, thread_id, TaskType::kCopyTask);
    auto stage_name = GetStageName(pipe_name);
    auto machine_id = stage_dag_->machine_id(stage_name);
    auto segment_name = stage_dag_->GetSegmentNameFromStageName(stage_name);
    auto succeeding_envelopes
      = segment_dag_->GetSucceedingDataNodeNames(segment_name);
    CHECK(succeeding_envelopes.size() >= 1);
    auto out_copy_data_name = build_envelope_name_from_envelopes(
      out_copy_envelope_prefix_, succeeding_envelopes, machine_id, local_id);
    auto out_copy_data_node = AddDataNode(out_copy_data_name);
    // Connect compute pipe <-> out_copy pipe, having a data node inbetween.
    std::vector<ONode*> inputs{ pipe_node };
    std::vector<ONode*> outputs{ out_copy_node };
    AddEdges(out_copy_data_node, inputs, outputs);

    // Track the mapping between in_copy and compute pipe
    copy_compute_map_.AddOutCopy(out_copy_name, pipe_name);
  }
}

template <typename Dtype>
void PipeDag<Dtype>::AddNetPipeNodes() {
  // For each pair of stages needing network communication, we insert a pair of
  // <out_net, in_net> pipe nodes for the connection.
  // For each stage in StageDag, insert in_net pipes node or out_net pipes
  // node if necessary. That means, a stage may have multiple in_net nodes or
  // multiple out_net nodes.
  DagIterator<StageDag<Dtype>, true> dag_iterator(*stage_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto stage_name = current_node->node_name();
    auto stage_node
      = dynamic_cast<const OpNode<StageMeta>*> (current_node);
    auto stage_meta = stage_node->op();
    auto stage_machine_id = stage_meta->machine_id();
    auto predecessors = stage_dag_->GetPrecedingOpNodeNames(stage_name);
    for (auto& predecessor : predecessors) {
      auto preceding_stage_node = stage_dag_->GetOpNode(predecessor);
      auto preceding_stage_name = preceding_stage_node->node_name();
      auto preceding_stage_meta = preceding_stage_node->op();
      auto preceding_machine_id = preceding_stage_meta->machine_id();
      if (preceding_machine_id != stage_machine_id) {
        // This stage need an in_net pipe to the preceding_stage
        AddNetPipeNode(true, preceding_stage_name, preceding_machine_id,
          stage_name, stage_machine_id);
      }
    }
    auto successors = stage_dag_->GetSucceedingOpNodeNames(stage_name);
    for (auto& successor : successors) {
      auto succeeding_stage_node = stage_dag_->GetOpNode(successor);
      auto succeeding_stage_name = succeeding_stage_node->node_name();
      auto succeeding_stage_meta = succeeding_stage_node->op();
      auto succeeding_machine_id = succeeding_stage_meta->machine_id();
      if (succeeding_machine_id != stage_machine_id) {
        // This stage need an out_net pipe to the succeeding_stage
        AddNetPipeNode(false, stage_name, stage_machine_id,
          succeeding_stage_name, succeeding_machine_id);
      }
    }
  }
}

template <typename Dtype>
void PipeDag<Dtype>::AddNetPipeNode(bool in, const std::string& from_stage_name,
  int32_t from_machine_id, const std::string& to_stage_name,
  int32_t to_machine_id) {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  auto local_id = id_map->net_thread_local_id();
  auto stage_machine_id = in ? to_machine_id : from_machine_id;
  auto thread_id
    = id_map->thread_id_from_machine_and_local(stage_machine_id, local_id);
  std::string net_prefix = in ? in_net_pipe_prefix_ : out_net_pipe_prefix_;
  auto net_pipe_name = build_net_pipe_name(
    net_prefix, from_stage_name, to_stage_name, from_machine_id, to_machine_id);
  auto net_pipe_node
    = AddOpNode(net_pipe_name, thread_id, TaskType::kNetTask);

  if (in) {
    stage_net_map_.AddInNet(from_stage_name, to_stage_name, net_pipe_name);
  } else {
    stage_net_map_.AddOutNet(from_stage_name, to_stage_name, net_pipe_name);
  }
}

template <typename Dtype>
void PipeDag<Dtype>::ConnectNetPipeNodes() {
  DagIterator<StageDag<Dtype>, true> dag_iterator(*stage_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto stage_name = current_node->node_name();
    auto stage_node = dynamic_cast<const OpNode<StageMeta>*> (current_node);
    auto stage_machine_id = stage_node->op()->machine_id();
    auto other_stage_in_net = stage_net_map_.GetOtherStageAndInNet(stage_name);
    if (other_stage_in_net.empty()) continue;

    // For each in_net, find its corresponding out_net and connect them
    // auto other_stage_in_net_nodes = in_net_node_it->second;
    for (auto& other_stage_to_in_net_pair : other_stage_in_net) {
      auto other_stage_name = other_stage_to_in_net_pair.first;
      auto in_net_name = other_stage_to_in_net_pair.second;
      auto in_net_node = GetOpNode(in_net_name);
      auto other_stage_out_net
        = stage_net_map_.GetOtherStageAndOutNet(other_stage_name);
      CHECK(!other_stage_out_net.empty());
      CHECK(other_stage_out_net.count(stage_name) > 0);
      auto out_net_name = other_stage_out_net[stage_name];
      auto out_net_node = GetOpNode(out_net_name);

      auto from_machine_id
        = stage_dag_->GetOpNode(other_stage_name)->op()->machine_id();
      auto to_machine_id
        = stage_dag_->GetOpNode(stage_name)->op()->machine_id();
      auto preceding_segment
        = stage_dag_->GetOpNode(other_stage_name)->op()->segment_name();
      auto segment
        = stage_dag_->GetOpNode(stage_name)->op()->segment_name();
      auto envelope_names
        = segment_dag_->FindDataNodesInBetween(preceding_segment, segment);
      CHECK(envelope_names.size() == 1);

      auto envelope_name = build_net_envelope_name(on_net_envelope_prefix_,
        envelope_names[0], from_machine_id, to_machine_id);
      auto envelope_node = AddDataNode(envelope_name);
      std::vector<ONode*> inputs{ out_net_node };
      std::vector<ONode*> outputs{ in_net_node };
      AddEdges(envelope_node, inputs, outputs);
    }
  }
}

template <typename Dtype>
void PipeDag<Dtype>::AddBoxingPipeNodes() {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  DagIterator<StageDag<Dtype>, true> dag_iterator(*stage_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto stage_name = current_node->node_name();
    auto stage_node = dynamic_cast<const OpNode<StageMeta>*>(current_node);
    auto stage_meta = stage_node->op();
    auto stage_machine_id = stage_meta->machine_id();
    auto segment_name = stage_dag_->GetSegmentNameFromStageName(stage_name);

    auto preceding_envelope_names
      = segment_dag_->GetPrecedingDataNodeNames(segment_name);
    auto succeeding_envelope_names
      = segment_dag_->GetSucceedingDataNodeNames(segment_name);
    CHECK(succeeding_envelope_names.size() >= 1);

    if (NeedInBoxingPipe(stage_name, stage_machine_id)) {
      auto boxing_node = AddBoxingPipeNode(true, stage_name, stage_machine_id);
      auto boxing_name = boxing_node->node_name();
      auto pipe_names = GetInDelegatePipesFromStage(stage_name);
      for (auto& pipe_name : pipe_names) {
        auto pipe_node = GetOpNode(pipe_name);
        auto local_id = GetThreadLocalId(pipe_name);
        auto pipe_envelope_name = build_boxing_envelope_name(
          in_boxing_envelope_prefix_, preceding_envelope_names,
          stage_machine_id, local_id);
        auto data_node = AddDataNode(pipe_envelope_name);
        std::vector<ONode*> inputs{ boxing_node };
        std::vector<ONode*> outputs{ pipe_node };
        AddEdges(data_node, inputs, outputs);
      }
      // Delegate the input of the stage to the in_boxing node
      stage_boxing_map_.AddStageAndInBoxing(stage_name, boxing_name);
    }
    if (NeedOutBoxingPipe(stage_name, stage_machine_id)) {
      auto boxing_node = AddBoxingPipeNode(false, stage_name, stage_machine_id);
      auto boxing_name = boxing_node->node_name();
      auto pipe_names = GetOutDelegatePipesFromStage(stage_name);
      for (auto& pipe_name : pipe_names) {
        auto pipe_node = GetOpNode(pipe_name);
        auto local_id = GetThreadLocalId(pipe_name);
        auto pipe_envelope_name = build_boxing_envelope_name(
          out_boxing_envelope_prefix_, succeeding_envelope_names,
          stage_machine_id, local_id);
        auto data_node = AddDataNode(pipe_envelope_name);
        std::vector<ONode*> inputs{ pipe_node };
        std::vector<ONode*> outputs{ boxing_node };
        AddEdges(data_node, inputs, outputs);
      }
      // Delegate the out of the stage to the out_boxing node
      stage_boxing_map_.AddStageAndOutBoxing(stage_name, boxing_name);
    }
  }
}

template <typename Dtype>
void PipeDag<Dtype>::AddBoxingInfos() {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  DagIterator<StageDag<Dtype>, true> dag_iterator(*stage_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto stage_name = current_node->node_name();
    if (stage_boxing_map_.HasInBoxing(stage_name)) {
      auto in_boxing_name = stage_boxing_map_.GetInBoxingFromStage(stage_name);
      AddInBoxingInfo(in_boxing_name, stage_name);
    }
    if (stage_boxing_map_.HasOutBoxing(stage_name)) {
      auto out_boxing_name = stage_boxing_map_.GetOutBoxingFromStage(stage_name);
      AddOutBoxingInfo(out_boxing_name, stage_name);
    }
  }
}

template <typename Dtype>
bool PipeDag<Dtype>::NeedInBoxingPipe(
  const std::string& current_stage,
  int32_t current_machine_id) {
  auto preceding_segments = GetPrecedingSegmentNames(current_stage);
  // No preceding segments, no in_boxing
  if (preceding_segments.size() == 0) return false;
  // More than one preceding segments, needs in_boxing
  if (preceding_segments.size() > 1) return true;
  // Come to here: a single preceding segment
  auto preceding_stages
    = stage_dag_->GetStageNamesFromSegmentName(preceding_segments[0]);
  CHECK(preceding_stages.size() >= 1);
  // More than one preceding stages (single preceding segment), needs in_boxing
  if (preceding_stages.size() > 1) return true;

  // Come to here: a single preceding stage
  int32_t preceding_machine_id = stage_dag_->machine_id(preceding_stages[0]);
  auto preceding_pipes = GetPipeNamesFromStageName(preceding_stages[0]);
  auto current_pipes = GetPipeNamesFromStageName(current_stage);
  if (preceding_machine_id != current_machine_id) {
    // Depends on a single other machine
    if (current_pipes.size() > 1) {
      // Current stage has multiple pipes
      return true;
    } else {
      // Depends on a single other machine, and current stage has a single pipe
      return false;
    }
  } else {
    // preceding_stage and current_stage are on the same machine, and
    // current_stage has the single depended stage.
    if (stage_boxing_map_.HasOutBoxing(preceding_stages[0])) {
      // The preceding_stage has an out_boxing pipe, which means the outputs of
      // the preceding_pipes will be aggregated into a single one even there are
      // multiple preceding_pipes. In this case, whether we need an in_boxing
      // pipe solely depends on the number of pipes of current_stage. Concretely,
      // we need an in_boxing pipe if there are more than one pipes in current
      // stage.
      if (current_pipes.size() > 1) {
        return true;
      } else {
        return false;
      }
    } else {
      // There is no out_boxing pipe in preceding_stage. In this case, we don't
      // need an in_boxing pipe if both the preceding_pipes and current_pipes
      // have only one pipe respectively. Otherwise, we need an in_boxing pipe
      // for current_stage, meanwhile, the in_boxing pipe is a pipe_to_pipe type.
      if (preceding_pipes.size() == 1 && current_pipes.size() == 1) {
        return false;
      } else {
        return true;
      }
    }
  }
}

template <typename Dtype>
bool PipeDag<Dtype>::NeedOutBoxingPipe(const std::string& current_stage,
  int32_t current_machine_id) {
  auto succeeding_segments = GetSucceedingSegmentNames(current_stage);
  // No succeeding segments, no out_boxing
  if (succeeding_segments.size() == 0) return false;
  // More than one succeeding segments, needs out_boxing
  if (succeeding_segments.size() > 1) return true;
  // Come to here: a single succeeding segment
  auto succeeding_stages
    = stage_dag_->GetStageNamesFromSegmentName(succeeding_segments[0]);
  CHECK(succeeding_stages.size() >= 1);
  // More than one succeeding stages (single succeeding segment), needs out_boxing
  if (succeeding_stages.size() > 1) return true;

  // Come to here: a single succeeding stage
  int32_t succeeding_machine_id = stage_dag_->machine_id(succeeding_stages[0]);
  auto succeeding_pipes = GetPipeNamesFromStageName(succeeding_stages[0]);
  auto current_pipes = GetPipeNamesFromStageName(current_stage);
  if (succeeding_machine_id != current_machine_id) {
    // Depend on a single other machine
    if (current_pipes.size() > 1) {
      // Current stage has multiple pipes
      return true;
    } else {
      // Depends on a single other machine, and current stage has a single pipe
      return false;
    }
  } else {
    // succeeding_stage and current_stage are on the same machine, and
    // current_stage has the single depended stage.
    auto preceding_of_succeeding_segments
      = GetPrecedingSegmentNames(succeeding_stages[0]);
    // We know there is at least one preceding stage for the
    // succeeding_stages[0], that is, the current_stage
    CHECK(preceding_of_succeeding_segments.size() >= 1);
    if (preceding_of_succeeding_segments.size() > 1) {
      // There must be an in_boxing pipe in succeeding_stages[0]
      if (current_pipes.size() > 1) {
        // current_pipes in current_stage must be aggregated into a single blob
        // to be consumed by the in_boxing pipe of succeeding_stages[0].
        return true;
      } else {
        return false;
      }
    } else {
      // There is only one preceding segment for the succeeding_stages[0]. We
      // will check how many stages in the preceding segment.
      auto preceding_stages_of_succeeding_segment
        = stage_dag_->GetStageNamesFromSegmentName(
        preceding_of_succeeding_segments[0]);
      if (preceding_stages_of_succeeding_segment.size() > 1) {
        // There must be an in_boxing pipe in succeeding_stages[0]
        if (current_pipes.size() > 1) {
          // current_pipes in current_stage must be aggregated into a single
          // blob to be consumed by the in_boxing pipe of succeeding_stages[0]
          return true;
        } else {
          return false;
        }
      } else {
        // The succeeding_stages[0] has only one preceding stage, that is, the
        // current_stage. At the same time, current stage has only one
        // succeeding_stage.
        // No matter whether we need a boxing pipe here, it will be handled by
        // the succeeding_stage. If needs, it will insert an in_boxing pipe here.
      }
    }
  }
}

template <typename Dtype>
std::vector<std::string> PipeDag<Dtype>::GetPrecedingSegmentNames(
  const std::string& stage_name) const {
  auto segment_name = stage_dag_->GetSegmentNameFromStageName(stage_name);
  return segment_dag_->GetPrecedingOpNodeNames(segment_name);
}

template <typename Dtype>
std::vector<std::string> PipeDag<Dtype>::GetSucceedingSegmentNames(
  const std::string& stage_name) const {
  auto segment_name = stage_dag_->GetSegmentNameFromStageName(stage_name);
  return segment_dag_->GetSucceedingOpNodeNames(segment_name);
}

template <typename Dtype>
std::vector<std::string> PipeDag<Dtype>::GetInDelegatePipesFromStage(
  const std::string& stage_name) const {
  return GetDelegatePipesFromStage(stage_name, true);
}

template <typename Dtype>
std::vector<std::string> PipeDag<Dtype>::GetOutDelegatePipesFromStage(
  const std::string& stage_name) const {
  return GetDelegatePipesFromStage(stage_name, false);
}

template <typename Dtype>
std::vector<std::string> PipeDag<Dtype>::GetDelegatePipesFromStage(
  const std::string& stage_name, bool in) const {
  auto pipes_in_stage = GetPipeNamesFromStageName(stage_name);
  std::vector<std::string> delegate_names;
  for (auto& pipe : pipes_in_stage) {
    if (in) {
      auto delegate_name = copy_compute_map_.GetInCopyIfHave(pipe);
      delegate_names.push_back(delegate_name);
    } else {
      auto delegate_name = copy_compute_map_.GetOutCopyIfHave(pipe);
      delegate_names.push_back(delegate_name);
    }
  }
  return delegate_names;
}

template <typename Dtype>
void PipeDag<Dtype>::AddInBoxingInfo(
  const std::string& in_boxing_name,
  const std::string& stage_name) {
  BoxingInfo boxing_info(true);
  auto delegate_in_pipes = GetInDelegatePipesFromStage(stage_name);

  auto segment_name = stage_dag_->GetSegmentNameFromStageName(stage_name);
  auto preceding_segment_names
    = segment_dag_->GetPrecedingOpNodeNames(segment_name);
  CHECK(preceding_segment_names.size() != 0);
  auto real_preceding_stage_names
    = stage_dag_->GetPrecedingOpNodeNames(stage_name);
  CHECK(real_preceding_stage_names.size() != 0);
  std::unordered_set<std::string> preceding_stage_set;
  for (auto& preceding_stage_name : real_preceding_stage_names) {
    preceding_stage_set.insert(preceding_stage_name);
  }

  for (auto& preceding_segment_name : preceding_segment_names) {
    SegmentSegmentPair segment_pair{ preceding_segment_name, segment_name };
    BoxingInfoElement boxing_info_elem;
    boxing_info_elem.SetOutputs(delegate_in_pipes);

    auto possible_preceding_stage_names
      = stage_dag_->GetStageNamesFromSegmentName(preceding_segment_name);
    std::vector<std::string> preceding_stage_names;
    for (auto& possible_preceding_stage_name : possible_preceding_stage_names) {
      if (preceding_stage_set.count(possible_preceding_stage_name) > 0) {
        preceding_stage_names.push_back(possible_preceding_stage_name);
      }
    }
    if (preceding_stage_names.size() > 1) {
      // Multiple preceding stages as inputs, needs to be updated to the direct
      // inputs. No matter whether the preceding stage and current stage are on
      // the same machine.
      boxing_info_elem.SetInputs(preceding_stage_names);
    } else {
      // Single preceding stage
      if (stage_boxing_map_.HasOutBoxing(preceding_stage_names[0])) {
        // The preceding stage has one out_boxing pipe, may update to the direct
        // inputs later. No matter whether the preceding_stage_names[0] and
        // stage_name are on the same machine, just set the preceding_stage_names
        // as inputs.
        boxing_info_elem.SetInputs(preceding_stage_names);
      } else {
        // The preceding stage has no out_boxing pipe, may update to the direct
        // inputs
        int32_t stage_machine_id = stage_dag_->machine_id(stage_name);
        int32_t preceding_machine_id
          = stage_dag_->machine_id(preceding_stage_names[0]);
        if (stage_machine_id != preceding_machine_id) {
          // Not on the same machine, just use preceding_stage_names[0] as input,
          // will update to the direct input later.
          boxing_info_elem.SetInputs(preceding_stage_names);
        } else {
          // On the same machine, a pipe_to_pipe connection, no need to update
          // direct input later
          auto delegate_out_pipes
            = GetOutDelegatePipesFromStage(preceding_stage_names[0]);
          // Use the pipes (delegate pipes, such as CopyD2H pipes) as inputs
          boxing_info_elem.SetInputs(delegate_out_pipes);
          boxing_info_elem.SetPipeToPipe(true);
        }
      }
    }
    boxing_info.AddSegmentPairBoxingInfo(segment_pair, boxing_info_elem);
  }
  boxing_info_map_.AddBoxingInfo(in_boxing_name, boxing_info);
}

template <typename Dtype>
void PipeDag<Dtype>::AddOutBoxingInfo(
  const std::string& out_boxing_name,
  const std::string& stage_name) {
  BoxingInfo boxing_info(false);
  auto delegate_out_pipes = GetOutDelegatePipesFromStage(stage_name);

  auto segment_name = stage_dag_->GetSegmentNameFromStageName(stage_name);
  auto succeeding_segment_names
    = segment_dag_->GetSucceedingOpNodeNames(segment_name);
  CHECK(succeeding_segment_names.size() != 0);
  auto real_succeeding_stage_names
    = stage_dag_->GetSucceedingOpNodeNames(stage_name);
  CHECK(real_succeeding_stage_names.size() != 0);
  std::unordered_set<std::string> succeeding_stage_set;
  for (auto& succeeding_stage_name : real_succeeding_stage_names) {
    succeeding_stage_set.insert(succeeding_stage_name);
  }

  for (auto& succeeding_segment_name : succeeding_segment_names) {
    SegmentSegmentPair segment_pair{ segment_name, succeeding_segment_name };
    BoxingInfoElement boxing_info_elem;
    boxing_info_elem.SetInputs(delegate_out_pipes);

    auto possible_succeeding_stage_names
      = stage_dag_->GetStageNamesFromSegmentName(succeeding_segment_name);
    std::vector<std::string> succeeding_stage_names;
    for (auto& possible_succeeding_stage_name : possible_succeeding_stage_names) {
      if (succeeding_stage_set.count(possible_succeeding_stage_name) > 0) {
        succeeding_stage_names.push_back(possible_succeeding_stage_name);
      }
    }

    if (succeeding_stage_names.size() > 1) {
      // Multiple succeeding stages as outputs, needs to be updated to the direct
      // outputs. No matter whether succeeding stage and current stage are on the
      // same machine.
      boxing_info_elem.SetOutputs(succeeding_stage_names);
    } else {
      // Single succeeding stage
      if (stage_boxing_map_.HasInBoxing(succeeding_stage_names[0])) {
        // The succeeding stage has one in_boxing pipe, use the succeeding stage
        // as output, may update later, no matter whether the succeeding and
        // current stage are on the same machine.
        boxing_info_elem.SetOutputs(succeeding_stage_names);
      } else {
        // The succeeding stage has no in_boxing pipe
        int32_t stage_machine_id = stage_dag_->machine_id(stage_name);
        int32_t succeeding_machine_id
          = stage_dag_->machine_id(succeeding_stage_names[0]);
        if (stage_machine_id != succeeding_machine_id) {
          // Not on the same machine, use the succeeding_stage_names[0] as input
          // will update later.
          boxing_info_elem.SetOutputs(succeeding_stage_names);
        } else {
          auto delegate_in_pipes
            = GetInDelegatePipesFromStage(succeeding_stage_names[0]);
          CHECK(delegate_in_pipes.size() == 1);
          // Use the pipes (delegate pipes, such as CopyH2D pipes) as outputs
          boxing_info_elem.SetOutputs(delegate_in_pipes);
          boxing_info_elem.SetPipeToPipe(true);
        }
      }
    }
    boxing_info.AddSegmentPairBoxingInfo(segment_pair, boxing_info_elem);
  }
  boxing_info_map_.AddBoxingInfo(out_boxing_name, boxing_info);
}

template <typename Dtype>
OpNode<PipeMeta>* PipeDag<Dtype>::AddBoxingPipeNode(
    bool is_in,
    const std::string& stage_name,
    int32_t stage_machine_id) {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  auto local_id = id_map->boxing_thread_local_id();
  auto thread_id
    = id_map->thread_id_from_machine_and_local(stage_machine_id, local_id);
  auto boxing_prefix = is_in ? in_boxing_pipe_prefix_ : out_boxing_pipe_prefix_;
  auto boxing_name = build_pipe_name_with_local_id(
    boxing_prefix, stage_name, stage_machine_id, local_id);
  auto boxing_node = AddOpNode(boxing_name, thread_id, TaskType::kBoxingTask);
  return boxing_node;
}

template <typename Dtype>
void PipeDag<Dtype>::ConnectOutgoingPipeNodes() {
  DagIterator<StageDag<Dtype>, true> dag_iterator(*stage_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto stage_name = current_node->node_name();
    auto stage_node = dynamic_cast<const OpNode<StageMeta>*>(current_node);
    auto stage_machine_id
      = stage_node->op()->machine_id();
    auto segment_name = stage_node->op()->segment_name();
    auto successor_stage_names
      = stage_dag_->GetSucceedingOpNodeNames(stage_name);
    if (successor_stage_names.size() == 0) continue;
    for (auto& successor_stage_name : successor_stage_names) {
      auto successor_node = stage_dag_->GetOpNode(successor_stage_name);
      auto successor_machine_id = successor_node->op()->machine_id();
      auto successor_segment_name = successor_node->op()->segment_name();
      auto middle_data
        = stage_dag_->FindDataNodesInBetween(stage_name, successor_stage_name);
      CHECK(middle_data.size() == 1);
      if (successor_machine_id != stage_machine_id) {
        // The current and the succeeding stage are not at the same machine
        NotSameMachineOutgoing(
          stage_name, stage_machine_id, segment_name,
          successor_stage_name, successor_machine_id, successor_segment_name);
      } else {
        // The current and the succeeding stage are at the same machine
        SameMachineOutgoing(
          stage_name, stage_machine_id, segment_name,
          successor_stage_name, successor_machine_id, successor_segment_name);
      }
    }
  }
}

template <typename Dtype>
void PipeDag<Dtype>::NotSameMachineOutgoing(const std::string& stage_name,
  int32_t stage_machine_id,
  const std::string& segment_name,
  const std::string& successor_stage_name,
  int32_t successor_machine_id,
  const std::string& successor_segment_name) {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  // Get the data name's prefix
  auto envelope_names
    = segment_dag_->FindDataNodesInBetween(segment_name, successor_segment_name);
  CHECK(envelope_names.size() == 1);
  std::string envelope_name = envelope_names[0];

  auto other_stage_out_net = stage_net_map_.GetOtherStageAndOutNet(stage_name);
  CHECK(other_stage_out_net.count(successor_stage_name) > 0);
  auto out_net_name = other_stage_out_net[successor_stage_name];
  auto out_net_node = GetOpNode(out_net_name);

  if (!stage_boxing_map_.HasOutBoxing(stage_name)) {
    // No out_boxing for this stage, directly connect the pipes in this stage
    // to the out_net node of the residing stage.
    // Two conclusions: (1) this stage has a single pipe; (2) the successor_stage
    // is the only successor of this stage.
    auto delegate_pipe_names = GetOutDelegatePipesFromStage(stage_name);
    CHECK(delegate_pipe_names.size() == 1);
    auto delegate_name = delegate_pipe_names[0];
    auto pipe_node = GetOpNode(delegate_name);

    auto pipe_local_id = GetThreadLocalId(delegate_name);
    auto data_name = build_envelope_name(to_net_envelope_prefix_,
      envelope_name, stage_machine_id, pipe_local_id);
    auto data_node = AddDataNode(data_name);

    std::vector<ONode*> inputs{ pipe_node };
    std::vector<ONode*> outputs{ out_net_node };
    AddEdges(data_node, inputs, outputs);
  } else {
    // Having out_boxing for this stage, connect the out_boxing node of this
    // stage to the out_net nodes of the residing stage.
    auto boxing_name = stage_boxing_map_.GetOutBoxingFromStage(stage_name);
    auto boxing_node = GetOpNode(boxing_name);

    auto data_name = build_net_envelope_name(to_net_envelope_prefix_,
      envelope_name, stage_machine_id, successor_machine_id);
    auto data_node = AddDataNode(data_name);

    std::vector<ONode*> inputs{ boxing_node };
    std::vector<ONode*> outputs{ out_net_node };
    AddEdges(data_node, inputs, outputs);

    // Update the boxing_info of the out_boxing node
    auto& boxing_info = boxing_info_map_.GetBoxingInfo(boxing_name);
    SegmentSegmentPair segment_pair{ segment_name, successor_segment_name };
    auto& boxing_info_elem = boxing_info.GetBoxingInfoElement(segment_pair);
    boxing_info_elem.UpdateOutput(successor_stage_name, out_net_name);
  }
}

template <typename Dtype>
void PipeDag<Dtype>::SameMachineOutgoing(
  const std::string& stage_name,
  int32_t stage_machine_id,
  const std::string& segment_name,
  const std::string& successor_stage_name,
  int32_t successor_machine_id,
  const std::string& successor_segment_name) {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  // Get the data name's prefix
  auto envelope_names
    = segment_dag_->FindDataNodesInBetween(segment_name, successor_segment_name);
  CHECK(envelope_names.size() == 1);
  std::string envelope_name = envelope_names[0];

  if (!stage_boxing_map_.HasOutBoxing(stage_name)) {
    // No out_boxing node, directly connect the pipe in |stage_name| to the
    // successor_stage_name
    if (!stage_boxing_map_.HasInBoxing(successor_stage_name)) {
      // No out_boxing to stage_name
      // No in_boxing node to successor_stage_name, directly connect two pipes
      // Only possible for a single pipe to a single pipe
      auto delegate_pipe_names = GetOutDelegatePipesFromStage(stage_name);
      CHECK(delegate_pipe_names.size() == 1)
        << "Neither out_boxing, nor in_boxing, the preceding stage can only"
        << " have a sinle pipe";
      auto delegate_name = delegate_pipe_names[0];
      auto pipe_node = GetOpNode(delegate_name);

      auto succeeding_delegate_pipes
        = GetInDelegatePipesFromStage(successor_stage_name);
      CHECK(succeeding_delegate_pipes.size() == 1);
      auto succeeding_delegate_pipe = succeeding_delegate_pipes[0];
      auto succeeding_pipe_node = GetOpNode(succeeding_delegate_pipe);

      auto pipe_local_id = GetThreadLocalId(succeeding_delegate_pipe);
      auto data_name = build_envelope_name(
        "", envelope_name, stage_machine_id, pipe_local_id);
      auto data_node = AddDataNode(data_name);
      std::vector<ONode*> inputs{ pipe_node };
      std::vector<ONode*> outputs{ succeeding_pipe_node };
      AddEdges(data_node, inputs, outputs);
    } else {
      // No out_boxing to stage_name
      // Having in_boxing node for successor_stage, connect pipes to in_boxing.
      auto delegate_pipe_names = GetOutDelegatePipesFromStage(stage_name);
      auto in_boxing_name
        = stage_boxing_map_.GetInBoxingFromStage(successor_stage_name);
      auto boxing_node = GetOpNode(in_boxing_name);
      for (auto& delegate_pipe_name : delegate_pipe_names) {
        auto pipe_node = GetOpNode(delegate_pipe_name);
        auto pipe_local_id = GetThreadLocalId(delegate_pipe_name);
        auto data_name = build_envelope_name(
          "", envelope_name, stage_machine_id, pipe_local_id);
        auto data_node = AddDataNode(data_name);
        std::vector<ONode*> inputs{ pipe_node };
        std::vector<ONode*> outputs{ boxing_node };
        AddEdges(data_node, inputs, outputs);
      }
      // Update the inputs of in_boxing if necessary
      auto& boxing_info = boxing_info_map_.GetBoxingInfo(in_boxing_name);
      SegmentSegmentPair segment_pair{ segment_name, successor_segment_name };
      auto& boxing_info_elem = boxing_info.GetBoxingInfoElement(segment_pair);
      if (!boxing_info_elem.pipe_to_pipe()) {
        CHECK(delegate_pipe_names.size() == 1);
        boxing_info_elem.UpdateInput(stage_name, delegate_pipe_names[0]);
      }
    }
  } else {
    // Having out_boxing node, connect the out_boxing node to the
    // successor_stage_name
    if (!stage_boxing_map_.HasInBoxing(successor_stage_name)) {

      auto delegate_pipe_names
        = GetInDelegatePipesFromStage(successor_stage_name);
      auto out_boxing_name
        = stage_boxing_map_.GetOutBoxingFromStage(stage_name);
      auto boxing_node = GetOpNode(out_boxing_name);
      CHECK(delegate_pipe_names.size() == 1);
      auto delegate_pipe_name = delegate_pipe_names[0];
      auto pipe_node = GetOpNode(delegate_pipe_name);
      auto pipe_local_id = GetThreadLocalId(delegate_pipe_name);
      auto data_name = build_envelope_name(
        "", envelope_name, stage_machine_id, pipe_local_id);
      auto data_node = AddDataNode(data_name);
      std::vector<ONode*> inputs{ pipe_node };
      std::vector<ONode*> outputs{ boxing_node };
      AddEdges(data_node, inputs, outputs);

      auto& boxing_info = boxing_info_map_.GetBoxingInfo(out_boxing_name);
      SegmentSegmentPair segment_pair{ segment_name, successor_segment_name };
      auto& boxing_info_elem = boxing_info.GetBoxingInfoElement(segment_pair);
      CHECK(!boxing_info_elem.pipe_to_pipe());
      boxing_info_elem.UpdateOutput(successor_stage_name, delegate_pipe_name);
    } else {
      // Having in_boxing node for successor, connect out_boxing to in_boxing.
      auto in_boxing_name
        = stage_boxing_map_.GetInBoxingFromStage(successor_stage_name);
      auto out_boxing_name
        = stage_boxing_map_.GetOutBoxingFromStage(stage_name);
      auto in_boxing_node = GetOpNode(in_boxing_name);
      auto out_boxing_node = GetOpNode(out_boxing_name);
      auto data_name = envelope_name + "_m" + std::to_string(stage_machine_id);
      auto data_node = AddDataNode(data_name);
      std::vector<ONode*> inputs{ out_boxing_node };
      std::vector<ONode*> outputs{ in_boxing_node };
      AddEdges(data_node, inputs, outputs);

      // Update the outputs of out_boxing
      SegmentSegmentPair segment_pair{ segment_name, successor_segment_name };
      auto& out_boxing_info = boxing_info_map_.GetBoxingInfo(out_boxing_name);
      auto& out_boxing_info_elem
        = out_boxing_info.GetBoxingInfoElement(segment_pair);
      CHECK(!out_boxing_info_elem.pipe_to_pipe());
      out_boxing_info_elem.UpdateOutput(successor_stage_name, in_boxing_name);

      // Update the inputs of in_boxing
      auto& in_boxing_info = boxing_info_map_.GetBoxingInfo(in_boxing_name);
      auto& in_boxing_info_elem
        = in_boxing_info.GetBoxingInfoElement(segment_pair);
      CHECK(!in_boxing_info_elem.pipe_to_pipe());
      in_boxing_info_elem.UpdateInput(stage_name, out_boxing_name);
    }
  }
}

template <typename Dtype>
void PipeDag<Dtype>::ConnectIncomingPipeNodes() {
  DagIterator<StageDag<Dtype>, true> dag_iterator(*stage_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto stage_name = current_node->node_name();
    auto stage_node = dynamic_cast<const OpNode<StageMeta>*>(current_node);
    auto stage_machine_id = stage_node->op()->machine_id();
    auto segment_name = stage_node->op()->segment_name();
    auto predecessor_stage_names = stage_dag_->GetPrecedingOpNodeNames(stage_name);
    if (predecessor_stage_names.size() == 0) continue;
    for (auto& predecessor_stage_name : predecessor_stage_names) {
      auto predecessor_node = stage_dag_->GetOpNode(predecessor_stage_name);
      auto predecessor_machine_id = predecessor_node->op()->machine_id();
      auto predecessor_segment_name = predecessor_node->op()->segment_name();
      if (predecessor_machine_id != stage_machine_id) {
        // The current and the preceding stage are not at the same machine
        NotSameMachineIncoming(
          predecessor_stage_name, predecessor_machine_id, predecessor_segment_name,
          stage_name, stage_machine_id, segment_name);
      } else {
        // The current and the preceding stages are at the same machine
        // Do nothing, since the edges should have been handled by
        // SameMachineOutgoing
      }
    }
  }
}

template <typename Dtype>
void PipeDag<Dtype>::NotSameMachineIncoming(
  const std::string& predecessor_stage_name,
  int32_t predecessor_machine_id,
  const std::string& predecessor_segment_name,
  const std::string& stage_name,
  int32_t stage_machine_id,
  const std::string& segment_name) {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  // Get the data name's prefix
  auto envelope_names
    = segment_dag_->FindDataNodesInBetween(predecessor_segment_name, segment_name);
  CHECK(envelope_names.size() == 1);
  std::string envelope_name = envelope_names[0];

  auto other_stage_in_net = stage_net_map_.GetOtherStageAndInNet(stage_name);
  CHECK(other_stage_in_net.count(predecessor_stage_name) > 0);
  auto in_net_name = other_stage_in_net[predecessor_stage_name];
  auto in_net_node = GetOpNode(in_net_name);

  if (!stage_boxing_map_.HasInBoxing(stage_name)) {
    // No in_boxing for this stage, directly connect the pipes in this stage
    // to the in_net node of the residing stage.
    // Two conclusions: (1) this stage has a single pipe; (2) the predecessor_stage
    // is the only predecessor of this stage
    auto delegate_pipe_names = GetInDelegatePipesFromStage(stage_name);
    CHECK(delegate_pipe_names.size() == 1);
    auto delegate_pipe_name = delegate_pipe_names[0];
    auto pipe_node = GetOpNode(delegate_pipe_name);

    auto pipe_local_id = GetThreadLocalId(delegate_pipe_name);
    auto data_name = build_envelope_name(from_net_envelope_prefix_,
      envelope_name, stage_machine_id, pipe_local_id);
    auto data_node = AddDataNode(data_name);

    std::vector<ONode*> inputs{ in_net_node };
    std::vector<ONode*> outputs{ pipe_node };
    AddEdges(data_node, inputs, outputs);
  } else {
    // Having in_boxing node
    auto boxing_name = stage_boxing_map_.GetInBoxingFromStage(stage_name);
    auto boxing_node = GetOpNode(boxing_name);

    auto data_name = build_net_envelope_name(from_net_envelope_prefix_,
      envelope_name, predecessor_machine_id, stage_machine_id);
    auto data_node = AddDataNode(data_name);

    std::vector<ONode*> inputs{ in_net_node };
    std::vector<ONode*> outputs{ boxing_node };
    AddEdges(data_node, inputs, outputs);

    // Update the boxing_info of the in_boxing node
    auto& boxing_info = boxing_info_map_.GetBoxingInfo(boxing_name);
    SegmentSegmentPair segment_pair{ predecessor_segment_name, segment_name };
    auto& boxing_info_elem = boxing_info.GetBoxingInfoElement(segment_pair);
    boxing_info_elem.UpdateInput(predecessor_stage_name, in_net_name);
  }
}

template <typename Dtype>
void PipeDag<Dtype>::AddDataNodesWithoutSuccessors() {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  // Lookup the last data nodes of SegmentDag (i.e., the nodes precisely before
  // the End nodes)
  auto envelope_names_without_successor
    = segment_dag_->GetPreceedingDataNodeNamesOfEndNode();
  for (auto& envelope_name : envelope_names_without_successor) {
    // For each 'the last data node', get the segment node producing it
    auto data_node_predecessors
      = segment_dag_->GetPreceedingOpNodeNamesOfDataNode(envelope_name);
    CHECK(data_node_predecessors.size() == 1);
    // For each found segment node, get the corresponding stage nodes
    auto stage_names
      = stage_dag_->GetStageNamesFromSegmentName(data_node_predecessors[0]);
    for (auto& stage_name : stage_names) {
      // For each stage node, get the compute pipe nodes
      auto pipe_names = GetPipeNamesFromStageName(stage_name);
      auto machine_id = stage_dag_->machine_id(stage_name);
      for (auto& pipe_name : pipe_names) {
        auto pipe_node = GetOpNode(pipe_name);
        auto local_id = GetThreadLocalId(pipe_name);
        auto pipe_envelope_name
          = build_envelope_name(compute_envelope_prefix_, envelope_name,
          machine_id, local_id);
        auto data_node = AddDataNode(pipe_envelope_name);
        data_node->AddParent(pipe_node);
      }
    }
  }
}

template <typename Dtype>
bool PipeDag<Dtype>::IsComputePipe(const std::string& pipe_name) const {
  auto pipe_node = GetOpNode(pipe_name);
  return pipe_node->op()->task_type() == TaskType::kComputeTask;
}

template <typename Dtype>
OpNode<PipeMeta>* PipeDag<Dtype>::AddOpNode(const std::string& pipe_name,
  int32_t thread_id,
  TaskType type) {
  auto op_node = NewOpNode(pipe_name);
  auto& pipe_meta = op_node->mutable_op();
  pipe_meta = std::make_shared<PipeMeta>();
  pipe_meta->mutable_thread_id() = thread_id;
  pipe_meta->mutable_task_type() = type;
  auto it = op_name_to_node_.find(pipe_name);
  CHECK(it == op_name_to_node_.end()) << "Duplicate op_name: " << pipe_name;
  op_name_to_node_.insert({pipe_name, op_node});
  return op_node;
}

template <typename Dtype>
BoxingInfo PipeDag<Dtype>::GetBoxingInfo(
  const std::string& boxing_name) {
  return boxing_info_map_.GetBoxingInfo(boxing_name);
}

template <typename Dtype>
std::vector<std::string> PipeDag<Dtype>::GetPipeNamesFromStageName(
  const std::string& stage_name) const {
  return stage_pipe_map_.PipesFromStage(stage_name);
}

template <typename Dtype>
DataNode<EnvelopeMeta>* PipeDag<Dtype>::AddDataNode(
    const std::string& data_name) {
  auto data_node = NewDataNode(data_name);
  auto& envelope_meta = data_node->mutable_data();
  envelope_meta = std::make_shared<EnvelopeMeta>();
  // TODO(jiyuan): fill in the blob_names
  auto it = data_name_to_node_.find(data_name);
  CHECK(it == data_name_to_node_.end())
    << "Duplicate data_name: " << data_name;
  data_name_to_node_.insert({data_name, data_node});
  return data_node;
}

template <typename Dtype>
std::string PipeDag<Dtype>::GetStageName(const std::string& pipe_name) const {
  return stage_pipe_map_.StageFromPipe(pipe_name);
}

template <typename Dtype>
StageStagePair PipeDag<Dtype>::GetStagePairFromNetPipe(
  const std::string& net_pipe) const {
  return stage_net_map_.GetStagePairFromNetName(net_pipe);
}

template <typename Dtype>
std::string PipeDag<Dtype>::build_pipe_name_with_local_id(
  const std::string& prefix,
  const std::string& stage_name,
  int32_t machine_id, int32_t local_id) const {
  auto segment_name
    = stage_dag_->GetOpNode(stage_name)->op()->segment_name();
  auto pipe_name
    = prefix + segment_name
    + "_m" + std::to_string(machine_id)
    + "_d" + std::to_string(local_id);
  return pipe_name;
}

template <typename Dtype>
std::string PipeDag<Dtype>::build_net_pipe_name(
  const std::string& prefix,
  const std::string& from_stage_name,
  const std::string& to_stage_name,
  int32_t from_machine_id,
  int32_t to_machine_id) const {
  auto from_segment_name
    = stage_dag_->GetOpNode(from_stage_name)->op()->segment_name();
  auto to_segment_name
    = stage_dag_->GetOpNode(to_stage_name)->op()->segment_name();
  auto pipe_name
    = prefix + from_segment_name + "_" + to_segment_name
    + "_m" + std::to_string(from_machine_id)
    + "_c" + std::to_string(to_machine_id);
  return pipe_name;
}

template <typename Dtype>
std::string PipeDag<Dtype>::build_net_envelope_name(
  const std::string& prefix,
  const std::string& segment_envelope_name,
  int32_t from_machine_id,
  int32_t to_machine_id) const {
  auto envelope_name
    = prefix + segment_envelope_name + "_"
    + std::to_string(from_machine_id) + "x" + std::to_string(to_machine_id);
  return envelope_name;
}

template <typename Dtype>
std::string PipeDag<Dtype>::build_envelope_name(
  const std::string& prefix,
  const std::string& segment_envelope_name,
  int32_t machine_id,
  int32_t thread_local_id) const {
  auto pipe_envelope_name
    = prefix + segment_envelope_name
    + "_m" + std::to_string(machine_id)
    + "_d" + std::to_string(thread_local_id);
  return pipe_envelope_name;
}

template <typename Dtype>
std::string PipeDag<Dtype>::build_boxing_envelope_name(
  const std::string& prefix,
  const std::vector<std::string>& segment_envelope_names,
  int32_t machine_id,
  int32_t thread_local_id) const {
  std::string segment_envelope_name = "";
  for (auto envelope_name : segment_envelope_names) {
    if (segment_envelope_name == "") {
      segment_envelope_name = envelope_name;
    } else {
      segment_envelope_name += "_" + envelope_name;
    }
  }
  auto pipe_envelope_name
    = prefix + segment_envelope_name
    + "_m" + std::to_string(machine_id)
    + "_d" + std::to_string(thread_local_id);
  return pipe_envelope_name;
}

template <typename Dtype>
std::string PipeDag<Dtype>::build_envelope_name_from_envelopes(
  const std::string& prefix,
  const std::vector<std::string>& segment_envelope_names,
  int32_t machine_id,
  int32_t local_id) const {
  std::string segment_envelope_name = "";
  for (auto envelope_name : segment_envelope_names) {
    if (segment_envelope_name == "") {
      segment_envelope_name = envelope_name;
    } else {
      segment_envelope_name += "_" + envelope_name;
    }
  }
  auto pipe_envelope_name
    = prefix + segment_envelope_name
    + "_m" + std::to_string(machine_id)
    + "_d" + std::to_string(local_id);
  return pipe_envelope_name;
}

template <typename Dtype>
int32_t PipeDag<Dtype>::GetThreadLocalId(
  const std::string& pipe_name) const {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  auto pipe_node = GetOpNode(pipe_name);
  auto thread_id = pipe_node->op()->thread_id();
  return id_map->local_id_from_thread_id(thread_id);
}

template<typename Dtype>
void PipeDag<Dtype>::AddDataNodeInBetween(DagNode* node1, DagNode* node2,
  const std::string& data_node_name) {
  auto data_node = AddDataNode(data_node_name);
  data_node->AddParent(node1);
  node2->AddParent(data_node);
}


template <typename Dtype>
void PipeDag<Dtype>::StagePipeMap::AddStagePipe(const std::string& stage_name,
  const std::string& pipe_name) {
  CHECK(pipe_to_stage_.count(pipe_name) == 0);
  pipe_to_stage_.insert({ pipe_name, stage_name });
  auto pipes_it = stage_to_pipes_.find(stage_name);
  if (pipes_it == stage_to_pipes_.end()) {
    std::vector<std::string> pipes{ pipe_name };
    stage_to_pipes_.insert({ stage_name, pipes });
  } else {
    pipes_it->second.push_back(pipe_name);
  }
}

template <typename Dtype>
const std::string& PipeDag<Dtype>::StagePipeMap::StageFromPipe(
  const std::string& pipe_name) const {
  auto stage_it = pipe_to_stage_.find(pipe_name);
  CHECK(stage_it != pipe_to_stage_.end());
  return stage_it->second;
}

template <typename Dtype>
const std::vector<std::string>& PipeDag<Dtype>::StagePipeMap::PipesFromStage(
  const std::string& stage_name) const {
  auto pipes_it = stage_to_pipes_.find(stage_name);
  CHECK(pipes_it != stage_to_pipes_.end());
  return pipes_it->second;
}

template <typename Dtype>
void PipeDag<Dtype>::CopyComputeMap::AddInCopy(const std::string& in_copy_name,
  const std::string& compute_name) {
  CHECK(pipe_to_in_copy_.count(compute_name) == 0);
  CHECK(in_copy_to_pipe_.count(in_copy_name) == 0);
  pipe_to_in_copy_.insert({ compute_name, in_copy_name });
  in_copy_to_pipe_.insert({ in_copy_name, compute_name });
}

template <typename Dtype>
void PipeDag<Dtype>::CopyComputeMap::AddOutCopy(const std::string& out_copy_name,
  const std::string& compute_name) {
  CHECK(pipe_to_out_copy_.count(compute_name) == 0);
  CHECK(out_copy_to_pipe_.count(out_copy_name) == 0);
  pipe_to_out_copy_.insert({ compute_name, out_copy_name });
  out_copy_to_pipe_.insert({ out_copy_name, compute_name });
}

template <typename Dtype>
std::string PipeDag<Dtype>::CopyComputeMap::GetInCopyIfHave(
  const std::string& compute_name) const {
  auto in_copy_it = pipe_to_in_copy_.find(compute_name);
  if (in_copy_it != pipe_to_in_copy_.end()) {
    return in_copy_it->second;
  } else {
    return compute_name;
  }
}

template <typename Dtype>
std::string PipeDag<Dtype>::CopyComputeMap::GetOutCopyIfHave(
  const std::string& compute_name) const {
  auto out_copy_it = pipe_to_out_copy_.find(compute_name);
  if (out_copy_it != pipe_to_out_copy_.end()) {
    return out_copy_it->second;
  } else {
    return compute_name;
  }
}

template <typename Dtype>
std::string PipeDag<Dtype>::CopyComputeMap::GetComputeFromCopy(
  const std::string& copy_name) const {
  auto in_pipe_it = in_copy_to_pipe_.find(copy_name);
  if (in_pipe_it == in_copy_to_pipe_.end()) {
    auto out_pipe_it = out_copy_to_pipe_.find(copy_name);
    CHECK(out_pipe_it != out_copy_to_pipe_.end());
    return out_pipe_it->second;
  } else {
    return in_pipe_it->second;
  }
}

template <typename Dtype>
void PipeDag<Dtype>::StageNetMap::AddInNet(const std::string& from_stage,
  const std::string& to_stage, const std::string& in_net) {
  CHECK(net_to_stage_pair_.count(in_net) == 0);
  StageStagePair stage_pair(from_stage, to_stage);
  net_to_stage_pair_.insert({ in_net, stage_pair });
  AddStagePairNet(to_stage, from_stage, in_net, &stage_to_stage_in_net_);
}

template <typename Dtype>
void PipeDag<Dtype>::StageNetMap::AddOutNet(const std::string& from_stage,
  const std::string& to_stage, const std::string& out_net) {
  CHECK(net_to_stage_pair_.count(out_net) == 0);
  StageStagePair stage_pair(from_stage, to_stage);
  net_to_stage_pair_.insert({ out_net, stage_pair });
  AddStagePairNet(from_stage, to_stage, out_net, &stage_to_stage_out_net_);
}

template <typename Dtype>
void PipeDag<Dtype>::StageNetMap::AddStagePairNet(const std::string& A,
  const std::string& B, const std::string& net_name,
  StageStageNet* stage_stage_net) {
  auto stage_it = stage_stage_net->find(A);
  if (stage_it == stage_stage_net->end()) {
    // The key of |A| does not occur in |stage_stage_net| previously
    std::unordered_map<std::string, std::string> other_stage_name_to_net;
    other_stage_name_to_net.insert({ B, net_name });
    stage_stage_net->insert({ A, other_stage_name_to_net });
  } else {
    // The key of |A| already exists in |stage_stage_net|
    auto other_stage_it = stage_it->second.find(B);
    CHECK(other_stage_it == stage_it->second.end())
      << "The machine_id must not occur previously";
    stage_it->second.insert({ B, net_name });
  }
}

template <typename Dtype>
StageStagePair PipeDag<Dtype>::StageNetMap::GetStagePairFromNetName(
  const std::string& net_name) const {
  auto stage_pair_it = net_to_stage_pair_.find(net_name);
  CHECK(stage_pair_it != net_to_stage_pair_.end());
  return stage_pair_it->second;
}

template <typename Dtype>
std::unordered_map<std::string, std::string>
PipeDag<Dtype>::StageNetMap::GetOtherStageAndInNet(
  const std::string& stage_name) const {
  auto stage_in_net_it = stage_to_stage_in_net_.find(stage_name);
  if (stage_in_net_it == stage_to_stage_in_net_.end()) {
    return std::unordered_map<std::string, std::string>();
  } else {
    return stage_in_net_it->second;
  }
}

template <typename Dtype>
std::unordered_map<std::string, std::string>
PipeDag<Dtype>::StageNetMap::GetOtherStageAndOutNet(
  const std::string& stage_name) const {
  auto stage_out_net_it = stage_to_stage_out_net_.find(stage_name);
  if (stage_out_net_it == stage_to_stage_out_net_.end()) {
    return std::unordered_map<std::string, std::string>();
  } else {
    return stage_out_net_it->second;
  }
}

template <typename Dtype>
void PipeDag<Dtype>::StageBoxingMap::AddStageAndInBoxing(
  const std::string& stage, const std::string& in_boxing) {
  CHECK(stage_to_in_boxing_.count(stage) == 0);
  stage_to_in_boxing_.insert({ stage, in_boxing });
}

template <typename Dtype>
void PipeDag<Dtype>::StageBoxingMap::AddStageAndOutBoxing(
  const std::string& stage, const std::string& out_boxing) {
  CHECK(stage_to_out_boxing_.count(stage) == 0);
  stage_to_out_boxing_.insert({ stage, out_boxing });
}

template <typename Dtype>
bool PipeDag<Dtype>::StageBoxingMap::HasInBoxing(
  const std::string& stage) const {
  return stage_to_in_boxing_.count(stage) > 0;
}

template <typename Dtype>
std::string PipeDag<Dtype>::StageBoxingMap::GetInBoxingFromStage(
  const std::string& stage) const {
  CHECK(HasInBoxing(stage));
  auto boxing_it = stage_to_in_boxing_.find(stage);
  return boxing_it->second;
}

template <typename Dtype>
bool PipeDag<Dtype>::StageBoxingMap::HasOutBoxing(
  const std::string& stage) const {
  return stage_to_out_boxing_.count(stage) > 0;
}

template <typename Dtype>
std::string PipeDag<Dtype>::StageBoxingMap::GetOutBoxingFromStage(
  const std::string& stage) const {
  CHECK(HasOutBoxing(stage));
  auto boxing_it = stage_to_out_boxing_.find(stage);
  return boxing_it->second;
}

INSTANTIATE_CLASS(PipeDag);
}  // namespace oneflow
