#include "dag/actor_dag.h"
#include <set>
#include <string>
#include <vector>
#include <queue>
#include "common/common.h"
#include "context/one.h"
#include "context/id_map.h"
#include "dag/node_meta.h"
#include "dag/dag_iterator.h"
#include "dag/logical_dag.h"
#include "dag/segment_dag.h"
#include "dag/stage_dag.h"
#include "dag/pipe_dag.h"
#include "dag/task_dag.h"
#include "dag/node_meta.h"
#include "dag/dag_node.h"

namespace oneflow {
template <typename Dtype>
ActorDag<Dtype>::ActorDag(
    std::shared_ptr<LogicalDag<Dtype>> logical_dag,
    std::shared_ptr<SegmentDag<Dtype>> segment_dag,
    std::shared_ptr<StageDag<Dtype>> stage_dag,
    std::shared_ptr<PipeDag<Dtype>> pipe_dag,
    PathType path_type,
    const std::string& name)
  : Dag<EventMeta, ActorMeta>(path_type, name), logical_dag_(logical_dag), segment_dag_(segment_dag),
  stage_dag_(stage_dag), pipe_dag_(pipe_dag), has_BP_(false) {
  Build();
  PrintDag(name_, true);
}
template <typename Dtype>
void ActorDag<Dtype>::Build() {
  if (path_type_ == PathType::kDataPath) {
    // Infer what pipes in PipeDag need BP error signals
    InferHasBpForPipeDag();
  }

  // Build a forward Dag by cloning from PipeDag
  ForwardBuildDag();

  if (has_BP_) {
    // Build a backward Dag according to the forward Dag, and connect the two
    BackwardBuildDag();
  }

  AddStartAndEndNodes();
}

template <typename Dtype>
void ActorDag<Dtype>::InferHasBpForPipeDag() {
  DagIterator<PipeDag<Dtype>, false> dag_iterator(*pipe_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto pipe_name = current_node->node_name();
    auto pipe_node
      = dynamic_cast<OpNode<PipeMeta>*> (current_node);
    auto pipe_meta = pipe_node->op();
    bool predecessor_has_BP = false;
    auto pipe_predecessors = pipe_dag_->GetPrecedingOpNodeNames(pipe_name);
    for (auto& pipe_predecessor : pipe_predecessors) {
      auto pipe_predecessor_node = pipe_dag_->GetOpNode(pipe_predecessor);
      auto pipe_predecessor_meta = pipe_predecessor_node->op();
      if (pipe_predecessor_meta->has_BP()) {
        predecessor_has_BP = true;
        continue;
      }
    }
    if (!pipe_meta->has_BP() && predecessor_has_BP) {
      // If has_BP of current pipe is false, check whether one of its predecessors
      // needs BP. If yes, current pipe also needs BP.
      pipe_meta->mutable_has_BP() = true;
    }
    if (pipe_meta->has_BP() && !predecessor_has_BP) {
      // Current pipe needs BP, while no predecessor needs BP.
      last_pipes_in_bp_.insert(pipe_name);
    }
    if (pipe_meta->has_BP()) {
      // If one of pipes needs BP, the current actor_dag needs BP.
      this->has_BP_ = true;
    }
  }
}

template <typename Dtype>
void ActorDag<Dtype>::ForwardBuildDag() {
  ForwardAddActorNodes();
  ForwardConnectActorNodes();
}

template <typename Dtype>
void ActorDag<Dtype>::BackwardBuildDag() {
  BackwardAddActorNodes();
  ConnectTurningPointAtLossNode();
  BackwardConnectActorNodes();
}

template <typename Dtype>
void ActorDag<Dtype>::ForwardAddActorNodes() {
  // For each pipe node in PipeDag, create an actor node in ActorDag.
  auto&& id_map = oneflow::TheOne<Dtype>::id_map();
  DagIterator<PipeDag<Dtype>, true> dag_iterator(*pipe_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto pipe_name = current_node->node_name();
    auto pipe_node
      = dynamic_cast<const OpNode<PipeMeta>*> (current_node);
    auto pipe_meta = pipe_node->op();
    auto pipe_thread_id = pipe_meta->thread_id();
    auto task_local_id = id_map->new_task_local_id(pipe_thread_id);
    auto task_id = id_map->task_id_from_thread_id_and_task_local_id(
      pipe_thread_id, task_local_id);
    auto actor_name = build_actor_name(forward_prefix_, pipe_name);
    auto actor_node
      = AddOpNode(actor_name, task_id, pipe_node->op()->task_type());
    actor_pipe_map_.AddForwardActorPipe(actor_name, pipe_name);

    if (pipe_meta->task_type() == TaskType::kBoxingTask) {
      ForwardUpdateBoxingInfo(pipe_name);
    }
  }
}
template <typename Dtype>
void ActorDag<Dtype>::ForwardConnectActorNodes() {
  // Insert event between contiguous actor pair. Note that the loss envelope in
  // PipeDag has no corresponding event in ActorDag.
  DagIterator<PipeDag<Dtype>, true> dag_iterator(*pipe_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto pipe_name = current_node->node_name();
    auto actor_name = actor_pipe_map_.GetForwardFromPipe(pipe_name);
    auto compute_node = GetOpNode(actor_name);
    // Connect each <pipe_name, successor> pair
    auto pipe_successors = pipe_dag_->GetSucceedingOpNodeNames(pipe_name);
    for (auto& pipe_successor : pipe_successors) {
      auto middle_data
        = pipe_dag_->FindDataNodesInBetween(pipe_name, pipe_successor);
      CHECK_EQ(middle_data.size(), 1);
      std::string event_name = "";
      event_name = build_event_name(forward_prefix_, middle_data[0]);
      auto data_node = AddDataNode(event_name);
      auto actor_successor = actor_pipe_map_.GetForwardFromPipe(pipe_successor);
      auto actor_successor_node = GetOpNode(actor_successor);
      std::vector<ONode*> inputs{ compute_node };
      std::vector<ONode*> outputs{ actor_successor_node };
      AddEdges(data_node, inputs, outputs);
    }
  }
}

template <typename Dtype>
void ActorDag<Dtype>::BackwardAddActorNodes() {
  // In reverse topological order, traverse the PipeDag and add an actor node
  // for each pipe node if the pipe node is required in backward pass.
  auto&& id_map = oneflow::TheOne<Dtype>::id_map();
  DagReverseIterator<PipeDag<Dtype>, true> dag_iterator(*pipe_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto pipe_name = current_node->node_name();

    if (pipe_dag_->IsLastOpNode(current_node)) {
      // Collect loss pipes as turning points of forward-pass and backward-pass
      loss_pipes_.push_back(pipe_name);
    }

    auto pipe_node
      = dynamic_cast<const OpNode<PipeMeta>*> (current_node);
    auto pipe_meta = pipe_node->op();
    if (!pipe_meta->has_BP()) continue; // Skip the pipe that does not need BP
    auto pipe_thread_id = pipe_meta->thread_id();
    auto task_local_id = id_map->new_task_local_id(pipe_thread_id);
    auto task_id = id_map->task_id_from_thread_id_and_task_local_id(
      pipe_thread_id, task_local_id);
    auto actor_name = build_actor_name(backward_prefix_, pipe_name);
    auto actor_node
      = AddOpNode(actor_name, task_id, pipe_node->op()->task_type());

    actor_pipe_map_.AddBackwardActorPipe(actor_name, pipe_name);
  }
}

template <typename Dtype>
void ActorDag<Dtype>::ConnectTurningPointAtLossNode() {
  for (auto& loss_pipe : loss_pipes_) {
    // Get the forward and backward actors corresponding to this |loss_pipe|
    auto forward_actor_name = actor_pipe_map_.GetForwardFromPipe(loss_pipe);
    auto backward_actor_name = actor_pipe_map_.GetBackwardFromPipe(loss_pipe);

    auto forward_actor_node = GetOpNode(forward_actor_name);
    auto backward_actor_node = GetOpNode(backward_actor_name);

    // Connect the forward and backward actors
    auto event_name = loss_pipe + "/turning_point";
    auto data_node = AddDataNode(event_name);
    std::vector<ONode*> inputs{ forward_actor_node };
    std::vector<ONode*> outputs{ backward_actor_node };
    AddEdges(data_node, inputs, outputs);
  }
}

template <typename Dtype>
void ActorDag<Dtype>::BackwardConnectActorNodes() {
  DagReverseIterator<PipeDag<Dtype>, true> dag_iterator(*pipe_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto pipe_node
      = dynamic_cast<const OpNode<PipeMeta>*> (current_node);
    auto pipe_meta = pipe_node->op();
    // Skip the pipe node that does not need BP
    if (!pipe_meta->has_BP()) continue;

    auto pipe_name = current_node->node_name();
    std::string actor_name = actor_pipe_map_.GetBackwardFromPipe(pipe_name);
    auto compute_node = GetOpNode(actor_name);
    if (last_pipes_in_bp_.count(pipe_name) > 0) {
      std::string envelope_name
        = build_event_name(backward_prefix_, pipe_name + "_data");
      auto data_node = AddDataNode(envelope_name);
      data_node->AddParent(compute_node);
      continue;  // No need to check its predecessors in Pipe Dag any more
    }

    // Connect each <predecessor, pipe_name> pair
    auto pipe_predecessors = pipe_dag_->GetPrecedingOpNodeNames(pipe_name);
    for (auto& pipe_predecessor : pipe_predecessors) {
      // If its predecessor does not need BP, not connect this to predecessor.
      if (!pipe_dag_->GetOpNode(pipe_predecessor)->op()->has_BP()) continue;
      auto actor_predecessor
        = actor_pipe_map_.GetBackwardFromPipe(pipe_predecessor);

      auto middle_data
        = pipe_dag_->FindDataNodesInBetween(pipe_predecessor, pipe_name);
      CHECK_EQ(middle_data.size(), 1);
      std::string event_name 
        = build_event_name(backward_prefix_, middle_data[0]);
      auto data_node = AddDataNode(event_name);
      auto actor_predecessor_node = GetOpNode(actor_predecessor);
      std::vector<ONode*> inputs{ compute_node };
      std::vector<ONode*> outputs{ actor_predecessor_node };
      AddEdges(data_node, inputs, outputs);
    }
  }
}

template <typename Dtype>
bool ActorDag<Dtype>::HasOpNode(const std::string& actor_name) const {
  auto actor_it = op_name_to_node_.find(actor_name);
  return actor_it != op_name_to_node_.end();
}

template <typename Dtype>
void ActorDag<Dtype>::ForwardUpdateBoxingInfo(
  const std::string& boxing_pipe_name) {
  auto boxing_actor_name = build_actor_name(forward_prefix_, boxing_pipe_name);
  auto pipe_boxing_info = pipe_dag_->GetBoxingInfo(boxing_pipe_name);
  auto segment_pairs = pipe_boxing_info.GetSegmentPairs();

  BoxingInfo actor_boxing_info(pipe_boxing_info.is_in_boxing());
  for (auto segment_pair : segment_pairs) {
    auto pipe_boxing_info_elem
      = pipe_boxing_info.GetBoxingInfoElement(segment_pair);
    BoxingInfoElement actor_boxing_info_elem;
    auto pipe_inputs = pipe_boxing_info_elem.GetOrderedInputs();
    auto pipe_outputs = pipe_boxing_info_elem.GetOrderedOutputs();
    std::vector<std::string> actor_inputs;
    std::vector<std::string> actor_outputs;
    for (auto& pipe_input : pipe_inputs) {
      auto actor_input = build_actor_name(forward_prefix_, pipe_input);
      actor_inputs.push_back(actor_input);
    }
    for (auto& pipe_output : pipe_outputs) {
      auto actor_output = build_actor_name(forward_prefix_, pipe_output);
      actor_outputs.push_back(actor_output);
    }
    actor_boxing_info_elem.SetInputs(actor_inputs);
    actor_boxing_info_elem.SetOutputs(actor_outputs);
    actor_boxing_info.AddSegmentPairBoxingInfo(
      segment_pair, actor_boxing_info_elem);
  }
  boxing_info_map_.AddBoxingInfo(boxing_actor_name, actor_boxing_info);
}

template <typename Dtype>
BoxingInfo ActorDag<Dtype>::GetForwardBoxingInfo(
  const std::string& boxing_actor_name) {
  return boxing_info_map_.GetBoxingInfo(boxing_actor_name);
}

template <typename Dtype>
OpNode<ActorMeta>* ActorDag<Dtype>::AddOpNode(
    const std::string& actor_name,
    int32_t task_id,
    TaskType type) {
  auto op_node = NewOpNode(actor_name);
  auto&& actor_meta = op_node->mutable_op();
  actor_meta = std::make_shared<ActorMeta>();
  actor_meta->mutable_task_id() = task_id;
  actor_meta->mutable_task_type() = type;
  bool is_forward = oneflow::strings::StartsWith(actor_name, forward_prefix_);
  actor_meta->mutable_is_forward() = is_forward;
  auto it = op_name_to_node_.find(actor_name);
  CHECK(it == op_name_to_node_.end()) << "Duplicate op_name: " << actor_name;
  op_name_to_node_.insert({actor_name, op_node});
  return op_node;
}

template <typename Dtype>
DataNode<EventMeta>* ActorDag<Dtype>::AddDataNode(
    const std::string& data_name) {
  auto data_node = NewDataNode(data_name);
  auto&& event_meta = data_node->mutable_data();
  event_meta = std::make_shared<EventMeta>();
  auto it = data_name_to_node_.find(data_name);
  CHECK(it == data_name_to_node_.end())
    << "Duplicate data_name: " << data_name;
  data_name_to_node_.insert({data_name, data_node});
  return data_node;
}

template <typename Dtype>
std::string ActorDag<Dtype>::build_actor_name(const std::string& prefix,
  const std::string& pipe_name) const {
  return prefix + pipe_name;
}

template <typename Dtype>
std::string ActorDag<Dtype>::build_event_name(const std::string& prefix,
  const std::string& envelope_name) const {
  return prefix + envelope_name;
}

template <typename Dtype>
std::vector<std::string> ActorDag<Dtype>::GetLayerNamesFromActor(
  const std::string& actor_name) const {
  auto segment_name = GetSegmentNameFromActor(actor_name);
  auto segment_node = segment_dag_->GetOpNode(segment_name);
  auto layers_in_segment = segment_node->op()->layer_names();
  return layers_in_segment;
}

template <typename Dtype>
std::string ActorDag<Dtype>::GetSegmentNameFromActor(
  const std::string& actor_name) const {
  auto stage_name = GetStageNameFromActor(actor_name);
  auto stage_node = stage_dag_->GetOpNode(stage_name);
  auto segment_name = stage_node->op()->segment_name();
  return segment_name;
}

template <typename Dtype>
std::string ActorDag<Dtype>::GetStageNameFromActor(
  const std::string& actor_name) const {
  auto actor_node = GetOpNode(actor_name);
  auto actor_meta = actor_node->op();
  CHECK(actor_meta->task_type() == TaskType::kDataTask
    || actor_meta->task_type() == TaskType::kComputeTask)
    << "Only kDataTask or kComputeTask have corresponding layers in LogicalDag";
  auto pipe_name = GetPipeNameFromActor(actor_name);
  auto stage_name = pipe_dag_->GetStageName(pipe_name);
  return stage_name;
}

template <typename Dtype>
std::string ActorDag<Dtype>::GetPipeNameFromActor(
  const std::string& actor_name) const {
  return actor_pipe_map_.GetPipeFromActor(actor_name);
}

template <typename Dtype>
std::string ActorDag<Dtype>::GetForwardTaskName(
  const std::string& backward_task_name) const {
  return actor_pipe_map_.GetForwardFromBackward(backward_task_name);
}

template <typename Dtype>
std::string ActorDag<Dtype>::GetBackwardTaskName(
  const std::string& forward_task_name) const {
  return actor_pipe_map_.GetBackwardFromForward(forward_task_name);
}

template <typename Dtype>
int32_t ActorDag<Dtype>::GetTaskID(const std::string& actor_name) const {
  auto actor_node = GetOpNode(actor_name);
  return actor_node->op()->task_id();
}

template <typename Dtype>
std::string ActorDag<Dtype>::GetForwardActorFromPipe(
  const std::string& pipe) const {
  return actor_pipe_map_.GetForwardFromPipe(pipe);
}

template <typename Dtype>
std::string ActorDag<Dtype>::GetBackwardActorFromPipe(
  const std::string& pipe) const {
  CHECK(has_BP_);
  return actor_pipe_map_.GetBackwardFromPipe(pipe);
}

template <typename Dtype>
void ActorDag<Dtype>::ActorPipeMap::AddForwardActorPipe(
  const std::string& forward_actor,
  const std::string& pipe) {
  CHECK(forward_actor_to_pipe_.count(forward_actor) == 0);
  forward_actor_to_pipe_.insert({ forward_actor, pipe });
  pipe_to_forward_actor_.insert({ pipe, forward_actor });
}

template <typename Dtype>
void ActorDag<Dtype>::ActorPipeMap::AddBackwardActorPipe(
  const std::string& backward_actor,
  const std::string& pipe) {
  CHECK(backward_actor_to_pipe_.count(backward_actor) == 0);
  backward_actor_to_pipe_.insert({ backward_actor, pipe });
  pipe_to_backward_actor_.insert({ pipe, backward_actor });
}

template <typename Dtype>
std::string ActorDag<Dtype>::ActorPipeMap::GetPipeFromActor(
  const std::string& actor) const {
  auto forward_pipe_it = forward_actor_to_pipe_.find(actor);
  if (forward_pipe_it == forward_actor_to_pipe_.end()) {
    auto backward_pipe_it = backward_actor_to_pipe_.find(actor);
    CHECK(backward_pipe_it != backward_actor_to_pipe_.end());
    return backward_pipe_it->second;
  } else {
    return forward_pipe_it->second;
  }
}

template <typename Dtype>
std::string ActorDag<Dtype>::ActorPipeMap::GetBackwardFromForward(
  const std::string& forward_actor) const {
  auto pipe_it = forward_actor_to_pipe_.find(forward_actor);
  CHECK(pipe_it != forward_actor_to_pipe_.end());
  auto backward_it = pipe_to_backward_actor_.find(pipe_it->second);
  // A forward actor may not have a corresponding backward actor
  if (backward_it == pipe_to_backward_actor_.end()) {
    return "";
  } else {
    return backward_it->second;
  }
}

template <typename Dtype>
std::string ActorDag<Dtype>::ActorPipeMap::GetForwardFromBackward(
  const std::string& backward_actor) const {
  auto pipe_it = backward_actor_to_pipe_.find(backward_actor);
  CHECK(pipe_it != backward_actor_to_pipe_.end());
  auto forward_it = pipe_to_forward_actor_.find(pipe_it->second);
  CHECK(forward_it != pipe_to_forward_actor_.end());
  return forward_it->second;
}

template <typename Dtype>
std::string ActorDag<Dtype>::ActorPipeMap::GetForwardFromPipe(
  const std::string& pipe) const {
  auto forward_it = pipe_to_forward_actor_.find(pipe);
  CHECK(forward_it != pipe_to_forward_actor_.end());
  return forward_it->second;
}

template <typename Dtype>
std::string ActorDag<Dtype>::ActorPipeMap::GetBackwardFromPipe(
  const std::string& pipe) const {
  auto backward_it = pipe_to_backward_actor_.find(pipe);
  CHECK(backward_it != pipe_to_backward_actor_.end());
  return backward_it->second;
}

// Use BFS to find the first descendant kComputeTask Node
template <typename Dtype>
std::string ActorDag<Dtype>::GetFirstDescendantComputeNodeName(
  const std::string& op_name) const {
  auto op_node = GetOpNode(op_name);
  std::queue<int32_t> que;
  que.push(op_node->node_id());
  while (!que.empty()) {
    auto next_node_id = que.front();
    que.pop();
    auto next_node = GetNode(next_node_id);
    if (next_node->Type() != NodeType::kDataNode) {
      auto op_node = dynamic_cast<OpNode<ActorMeta>*>(next_node);
      auto actor_meta = op_node->op();
      if (actor_meta->task_type() == TaskType::kComputeTask) {
        return op_node->node_name();
      }
    }

    auto successors = next_node->successors();
    if (successors.size() > 0) {
      for (auto successor : successors) {
         que.push(successor);
      }
    }
  }

  return "";
}

INSTANTIATE_CLASS(ActorDag);

}  // namespace oneflow
