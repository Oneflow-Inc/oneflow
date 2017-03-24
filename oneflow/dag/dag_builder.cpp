#include "dag/dag_builder.h"
#include "dag/logical_dag.h"
#include "dag/placement_group_dag.h"
#include "dag/segment_dag.h"
#include "dag/stage_dag.h"
#include "dag/pipe_dag.h"
#include "dag/actor_dag.h"
#include "dag/dag_iterator.h"
#include "dag/node_meta.h"
#include "dag/dag_node.h"
#include "dag/boxing_task_dag.h"
#include "dag/compute_task_dag.h"
#include "dag/copy_task_dag.h"
#include "dag/net_task_dag.h"
#include "common/common.h"
#include "path/base_path.h"
#include "path/path_type.h"

namespace oneflow {
template <typename Dtype>
DagBuilder<Dtype>::DagBuilder(const std::string& net_name,
  BasePath<Dtype>* path,
  std::shared_ptr<NetDescriptor> net_descriptor,
  std::shared_ptr<StrategyDescriptor> strategy_descriptor) : net_name_(net_name),
  path_(path),
  net_descriptor_(net_descriptor), strategy_descriptor_(strategy_descriptor) {
}

template <typename Dtype>
DagBuilder<Dtype>::~DagBuilder() {}

template <typename Dtype>
std::shared_ptr<LogicalDag<Dtype>> DagBuilder<Dtype>::logical_dag() const {
  return logical_dag_;
}

template <typename Dtype>
std::shared_ptr<SegmentDag<Dtype>> DagBuilder<Dtype>::segment_dag() const {
  return segment_dag_;
}

template <typename Dtype>
std::shared_ptr<StageDag<Dtype>> DagBuilder<Dtype>::stage_dag() const {
  return stage_dag_;
}

template <typename Dtype>
std::shared_ptr<PipeDag<Dtype>> DagBuilder<Dtype>::pipe_dag() const {
  return pipe_dag_;
}

template <typename Dtype>
std::shared_ptr<ActorDag<Dtype>> DagBuilder<Dtype>::actor_dag() const {
  return actor_dag_;
}

template <typename Dtype>
const SegmentTaskMap& DagBuilder<Dtype>::segment_task_map() const {
  return segment_task_map_;
}

template <typename Dtype>
bool DagBuilder<Dtype>::has_BP() const {
  return actor_dag_->has_BP();
}

template <typename Dtype>
void DagBuilder<Dtype>::Build() {
  // Generate {LogicalDag, PlacementGroupDag} according to NetDescriptor and
  // StrategyDescriptor. And then, generate SegmentDag, StageDag, PipeDag, and
  // finally ActorDag.
  BuildActorDag();

  // Generate a TaskDag for each actor in ActorDag, and build each TaskDag.
  BuildTaskDags();

  // Add the produced RegisterInfos of each TaskDag.
  AddProducedRegisterInfos();

  // For each TaskDag, add its consumed RegisterInfos inside the same path.
  AddConsumedRegisterInfosInPath();

  // Memorize the segment<->task correspondences, useful for path sharing.
  BuildSegmentTaskMap();
}

template <typename Dtype>
void DagBuilder<Dtype>::Setup() {
  SetupTaskDags();
}

template <typename Dtype>
void DagBuilder<Dtype>::BuildActorDag() {
  PathType path_type = path_->path_type();
  logical_dag_.reset(new LogicalDag<Dtype>(
    net_descriptor_, path_type, net_name_ + "_logical_dag"));
  placement_group_dag_.reset(new PlacementGroupDag<Dtype>(
    logical_dag_, strategy_descriptor_, path_type,
    net_name_ + "_placement_group_dag"));
  segment_dag_.reset(new SegmentDag<Dtype>(
    logical_dag_, path_type, net_name_ + "_segment_dag"));
  segment_dag_->Build();
  stage_dag_.reset(new StageDag<Dtype>(
    logical_dag_,
    segment_dag_, path_type, net_name_ + "_stage_dag"));
  pipe_dag_.reset(new PipeDag<Dtype>(
    segment_dag_,
    stage_dag_, path_type, net_name_ + "_pipe_dag"));
  actor_dag_.reset(new ActorDag<Dtype>(
    logical_dag_,
    segment_dag_,
    stage_dag_,
    pipe_dag_, path_type, net_name_ + "_actor_dag"));
}

template <typename Dtype>
void DagBuilder<Dtype>::BuildTaskDags() {
  // 1, Add all the empty TaskDag objects
  DagIterator<ActorDag<Dtype>, true> dag_iterator(*actor_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto actor_node =
      dynamic_cast<const OpNode<ActorMeta>*>(current_node);
    CHECK_NOTNULL(actor_node);
    auto actor_meta = actor_node->op();
    int32_t task_id = actor_meta->task_id();
    auto task_name = actor_node->node_name();  // task_name is the actor_name
    auto task_type = actor_meta->task_type();
    bool is_forward = actor_meta->is_forward();
    AddTaskDag(
      task_id, task_name, task_type, is_forward, actor_dag_->path_type());
  }

  // 2,  Building forward TaskDags has to follow a specific order of task types
  // due to the dependencies among TaskDags.
  BuildForwardTaskDags();

  // 3, Building the backward TaskDags has no special ordering requirement, just
  // follow the topologically-sorted order.
  if (actor_dag_->has_BP()) {
    BuildBackwardTaskDags();
  }
}

template <typename Dtype>
void DagBuilder<Dtype>::AddProducedRegisterInfos() {
  DagIterator<ActorDag<Dtype>, true> dag_iterator(*actor_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto actor_node =
      dynamic_cast<const OpNode<ActorMeta>*>(current_node);
    CHECK_NOTNULL(actor_node);
    auto actor_meta = actor_node->op();
    int32_t task_id = actor_meta->task_id();
    auto task_dag = GetTaskDag(task_id);
    task_dag->AddProducedRegisterInfos();
  }
}

template <typename Dtype>
void DagBuilder<Dtype>::AddConsumedRegisterInfosInPath() {
  DagIterator<ActorDag<Dtype>, true> dag_iterator(*actor_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto actor_node =
      dynamic_cast<const OpNode<ActorMeta>*>(current_node);
    CHECK_NOTNULL(actor_node);
    auto actor_meta = actor_node->op();
    int32_t task_id = actor_meta->task_id();
    auto task_dag = GetTaskDag(task_id);
    task_dag->AddConsumedRegisterInfosInPath();
  }
}

template <typename Dtype>
void DagBuilder<Dtype>::AddTaskDag(int32_t task_id,
  const std::string& task_name, TaskType task_type,
  bool is_forward, PathType path_type) {
  // Create concrete TaskDag according to its type
  std::shared_ptr<TaskDag<Dtype>> task_dag;
  switch (task_type) {
  case TaskType::kDataTask:
    task_dag.reset(new ComputeTaskDag<Dtype>(
      *this, task_type, task_id, path_type, task_name, is_forward));
    break;
  case TaskType::kCopyTask:
    task_dag.reset(new CopyTaskDag<Dtype>(
      *this, task_type, task_id, path_type, task_name, is_forward));
    break;
  case TaskType::kComputeTask:
    task_dag.reset(new ComputeTaskDag<Dtype>(
      *this, task_type, task_id, path_type, task_name, is_forward));
    break;
  case TaskType::kBoxingTask:
    task_dag.reset(new BoxingTaskDag<Dtype>(
      *this, task_type, task_id, path_type, task_name, is_forward));
    break;
  case TaskType::kNetTask:
    task_dag.reset(new NetTaskDag<Dtype>(
      *this, task_type, task_id, path_type, task_name, is_forward));
    break;
  default:
    LOG(FATAL) << "Unknown task type";
    break;
  }
  ordered_task_id_to_dag_.insert({ task_id, task_dag });
  unordered_task_id_to_dag_.insert({ task_id, task_dag });
  task_name_to_dag_.insert({ task_name, task_dag });

  auto&& id_map = oneflow::TheOne<Dtype>::id_map();
  int32_t thread_id = id_map->thread_id_from_task_id(task_id);
  int32_t machine_id = id_map->machine_id_from_thread_id(thread_id);
  int32_t thread_local_id = id_map->thread_local_id_from_task_id(task_id);
  int32_t task_local_id = id_map->task_local_id_from_task_id(task_id);

  auto task_dags_of_machine_it = all_task_dags_.find(machine_id);
  if (task_dags_of_machine_it == all_task_dags_.end()) {
    TaskDagsOfMachine task_dags_of_machine;
    task_dags_of_machine.insert({ thread_local_id, { task_dag } });
    all_task_dags_.insert({ machine_id, task_dags_of_machine });
  } else {
    auto task_dags_of_thread_it
      = task_dags_of_machine_it->second.find(thread_local_id);
    if (task_dags_of_thread_it == task_dags_of_machine_it->second.end()) {
      task_dags_of_machine_it->second.insert({ thread_local_id, { task_dag } });
    } else {
      task_dags_of_thread_it->second.push_back(task_dag);
    }
  }

  auto alias_task_id = id_map->task_id_from_thread_id_and_task_local_id(
    thread_id,
    task_local_id);
  CHECK_EQ(task_id, alias_task_id);
}

template <typename Dtype>
void DagBuilder<Dtype>::BuildForwardTaskDags() {
  // Build TaskDag for each actor. There exist dependencies among TaskDags.
  // For example, the copy task depends on the compute task it serves to
  // to know the blobs needed to be copied. Therefore, we follow a specific
  // order to build TaskDag for actors.
  BuildForwardTaskDagsOfType(TaskType::kDataTask);
  BuildForwardTaskDagsOfType(TaskType::kComputeTask);
  BuildForwardTaskDagsOfType(TaskType::kCopyTask);
  BuildForwardTaskDagsOfType(TaskType::kBoxingTask);
  BuildForwardTaskDagsOfType(TaskType::kNetTask);
}

template <typename Dtype>
void DagBuilder<Dtype>::BuildForwardTaskDagsOfType(TaskType expected_task_type) {
  DagIterator<ActorDag<Dtype>, true> dag_iterator(*actor_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto actor_node =
      dynamic_cast<const OpNode<ActorMeta>*>(current_node);
    CHECK_NOTNULL(actor_node);
    auto actor_meta = actor_node->op();
    auto task_type = actor_meta->task_type();
    if (task_type != expected_task_type) continue;  // Only care about |task_type|
    if (!actor_meta->is_forward()) continue;    // Only care about forward tasks
    int32_t task_id = actor_meta->task_id();
    auto task_dag = GetTaskDag(task_id);
    task_dag->Build();

    auto actor_name = current_node->node_name();
    if (task_type == TaskType::kComputeTask
		|| task_type == TaskType::kCopyTask
      || task_type == TaskType::kBoxingTask
      || task_type == TaskType::kNetTask) {
      task_dag->PrintDag(actor_name, true, true);
    }
  }
}

template <typename Dtype>
void DagBuilder<Dtype>::BuildBackwardTaskDags() {
  DagIterator<ActorDag<Dtype>, true> dag_iterator(*actor_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto actor_node =
      dynamic_cast<const OpNode<ActorMeta>*>(current_node);
    CHECK_NOTNULL(actor_node);
    auto actor_meta = actor_node->op();
    // Only care about backward TaskDags
    if (actor_meta->is_forward()) continue;
    auto actor_name = current_node->node_name();
    int32_t task_id = actor_meta->task_id();
    auto task_dag = GetTaskDag(task_id);
    task_dag->Build();

    auto task_type = actor_meta->task_type();
    if (task_type == TaskType::kComputeTask
		|| task_type == TaskType::kCopyTask
      || task_type == TaskType::kBoxingTask
      || task_type == TaskType::kNetTask) {
      task_dag->PrintDag(actor_name, true, true);
    }
  }
}

template <typename Dtype>
void DagBuilder<Dtype>::SetupTaskDags() {
  DagIterator<ActorDag<Dtype>, true> dag_iterator(*actor_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto actor_node =
      dynamic_cast<const OpNode<ActorMeta>*>(current_node);
    CHECK_NOTNULL(actor_node);
    auto actor_meta = actor_node->op();
    int32_t task_id = actor_meta->task_id();
    auto task_dag = GetTaskDag(task_id);
    task_dag->Setup();
  }
}

template <typename Dtype>
void DagBuilder<Dtype>::BuildSegmentTaskMap() {
  auto&& id_map = oneflow::TheOne<Dtype>::id_map();
  auto compute_segments = segment_dag_->GetComputeSegments();
  for (auto& compute_segment : compute_segments) {
    auto device_set = segment_dag_->DeviceSetOfSegment(compute_segment);
    auto stages = stage_dag_->GetStageNamesFromSegmentName(compute_segment);
    for (auto& stage : stages) {
      auto pipes = pipe_dag_->GetPipeNamesFromStageName(stage);
      for (auto& pipe : pipes) {
        auto forward_actor = actor_dag_->GetForwardActorFromPipe(pipe);
        auto forward_task_id = actor_dag_->GetTaskID(forward_actor);
        auto forward_device_id = id_map->thread_id_from_task_id(forward_task_id);
        segment_task_map_.AddSegmentForwardTaskPair(compute_segment,
          forward_device_id, forward_actor);
        if (has_BP()) {
          auto backward_actor = actor_dag_->GetBackwardActorFromPipe(pipe);
          auto backward_task_id = actor_dag_->GetTaskID(backward_actor);
          auto backward_device_id
            = id_map->thread_id_from_task_id(backward_task_id);
          segment_task_map_.AddSegmentBackwardTaskPair(compute_segment,
            backward_device_id, backward_actor);
        }
      }
    }
  }
}

template <typename Dtype>
std::shared_ptr<TaskDag<Dtype>> DagBuilder<Dtype>::GetTaskDag(
  int32_t task_id) const {
  auto task_dag_it = unordered_task_id_to_dag_.find(task_id);
  CHECK(task_dag_it != unordered_task_id_to_dag_.end());
  return task_dag_it->second;
};

template <typename Dtype>
std::string DagBuilder<Dtype>::GetTaskName(const std::string& segment_name,
  bool is_forward, int32_t device_id) const {
  if (is_forward) {
    return segment_task_map_.GetForwardTask(segment_name, device_id);
  } else {
    return segment_task_map_.GetBackwardTask(segment_name, device_id);
  }
}

template <typename Dtype>
std::shared_ptr<TaskDag<Dtype>> DagBuilder<Dtype>::GetTaskDagByName(
  const std::string& task_name) const {
  auto task_id = actor_dag_->GetTaskID(task_name);
  return GetTaskDag(task_id);
}

template <typename Dtype>
std::vector<int32_t> DagBuilder<Dtype>::GetDeviceIDs(
  const std::string& segment_name) const {
  return segment_task_map_.GetDeviceIDs(segment_name);
}

template <typename Dtype>
std::shared_ptr<TaskDag<Dtype>> DagBuilder<Dtype>::GetCrossPathTaskDag(
  const PathSharingDetail& sharing_detail, int32_t device_id) const {
  bool is_forward = sharing_detail.task_direction == TaskDirection::kForward;
  bool is_placeholder = sharing_detail.task_placeholder == TaskPlaceholder::kYes;
  auto task_name = GetTaskName(
    sharing_detail.segment_name, is_forward, device_id);
  if (!is_placeholder) {
    return GetTaskDagByName(task_name);
  } else if (sharing_detail.role == PathSharingRole::kConsumer) {
    auto task_dag = GetTaskDagByName(task_name);
    // task_dag->SetIsPlaceholder(true);

    auto consumers = task_dag->GetImmediateConsumerNamesInPath();
    CHECK_EQ(consumers.size(), 1);
    return GetTaskDagByName(consumers[0]);
  } else if (sharing_detail.role == PathSharingRole::kProducer) {
    auto task_dag = GetTaskDagByName(task_name);
    // task_dag->SetIsPlaceholder(true);

    auto producers = task_dag->GetImmediateProducerNamesInPath();
    CHECK_EQ(producers.size(), 1);
    return GetTaskDagByName(producers[0]);
  }
}
INSTANTIATE_CLASS(DagBuilder);
}  // namespace oneflow
