#include "task/job_manager.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <limits>
#include "common/common.h"
#include "dag/actor_dag.h"
#include "dag/node_meta.h"
#include "dag/copy_task_dag.h"
#include "dag/compute_task_dag.h"
#include "dag/boxing_task_dag.h"
#include "dag/net_task_dag.h"
#include "dag/dag_iterator.h"
#include "context/one.h"
#include "context/id_map.h"
#include "context/config_parser.h"
#include "context/machine_descriptor.h"
#include "context/strategy_descriptor.h"
#include "device/device_descriptor.h"

namespace oneflow {

template <typename Dtype>
JobManager<Dtype>::JobManager(
  std::shared_ptr<ActorDag<Dtype>> actor_dag,
  std::shared_ptr<ActorDag<Dtype>> ps_actor_dag)
  : actor_dag_(actor_dag), ps_actor_dag_(ps_actor_dag),
  ps_actor_dag_task_setup_(false) {
}
template <typename Dtype>
JobManager<Dtype>::~JobManager() {}

template <typename Dtype>
void JobManager<Dtype>::Init() {
  ProcessAllTaskDags();
  GetMemoryQuota();
  SetMemoryPolicy();
}
template <typename Dtype>
void JobManager<Dtype>::ProcessAllTaskDags() {
  CollectTaskDags(actor_dag_);
  BuildTaskDags(actor_dag_);
  SetupTaskDags(actor_dag_);
  if (ps_actor_dag_) {
    CollectTaskDags(ps_actor_dag_);
    BuildTaskDags(ps_actor_dag_);
    SetupTaskDags(ps_actor_dag_);
  }
}
template <typename Dtype>
void JobManager<Dtype>::CollectTaskDags(
  std::shared_ptr<ActorDag<Dtype>> actor_dag) {
  //auto& id_map = oneflow::TheOne<Dtype>::id_map();
  //bool is_ps_dag = actor_dag->IsPSDag();
  //DagIterator<ActorDag<Dtype>, true> dag_iterator(*actor_dag);
  //for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
  //  auto current_node = dag_iterator.CurrentNode();
  //  if (current_node->Type() != NodeType::kOpNode) continue;
  //  auto actor_node =
  //    dynamic_cast<const OpNode<ActorMeta>*>(current_node);
  //  CHECK_NOTNULL(actor_node);
  //  auto actor_meta = actor_node->op();
  //  int32_t task_id = actor_meta->task_id();
  //  auto task_name = actor_node->node_name();  // task_name is the actor_name
  //  auto task_type = actor_meta->task_type();
  //  bool is_forward = actor_meta->is_forward();
  //  AddTaskDag(task_id, task_name, task_type, is_forward, is_ps_dag,
  //    actor_dag->path_type());
  //}
}
template <typename Dtype>
void JobManager<Dtype>::AddTaskDag(int32_t task_id,
  const std::string& task_name, TaskType task_type, bool is_forward,
  bool is_ps_dag, PathType path_type) {
  // Create concrete TaskDag according to its type
  std::shared_ptr<TaskDag<Dtype>> task_dag;
  //switch (task_type) {
  //case TaskType::kDataTask:
  //  task_dag.reset(new ComputeTaskDag<Dtype>(
  //    *this, task_type, task_id, path_type, task_name, is_forward));
  //  break;
  //case TaskType::kCopyTask:
  //  task_dag.reset(new CopyTaskDag<Dtype>(
  //    *this, task_type, task_id, path_type, task_name, is_forward));
  //  break;
  //case TaskType::kComputeTask:
  //  task_dag.reset(new ComputeTaskDag<Dtype>(
  //    *this, task_type, task_id, path_type, task_name, is_forward));
  //  break;
  //case TaskType::kBoxingTask:
  //  task_dag.reset(new BoxingTaskDag<Dtype>(
  //    *this, task_type, task_id, path_type, task_name, is_forward));
  //  break;
  //case TaskType::kNetTask:
  //  task_dag.reset(new NetTaskDag<Dtype>(
  //    *this, task_type, task_id, path_type, task_name, is_forward));
  //  break;
  //default:
  //  LOG(FATAL) << "Unknown task type";
  //  break;
  //}
  // FIXME(jiyuan): new task_dag
  ordered_task_id_to_dag_.insert({ task_id, task_dag });
  unordered_task_id_to_dag_.insert({ task_id, task_dag });
  task_name_to_dag_.insert({ task_name, task_dag });

  auto& id_map = oneflow::TheOne<Dtype>::id_map();
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
void JobManager<Dtype>::BuildTaskDags(
  std::shared_ptr<ActorDag<Dtype>> actor_dag) {
  BuildForwardTaskDags(actor_dag);
  if (actor_dag->has_BP()) {
    BuildBackwardTaskDags(actor_dag);
  }
}
template <typename Dtype>
void JobManager<Dtype>::BuildForwardTaskDags(
  std::shared_ptr<ActorDag<Dtype>> actor_dag) {
  // Build TaskDag for each actor. There exist dependencies among TaskDags.
  // For example, the copy task depends on the compute task it serves to
  // to know the blobs needed to be copied. Therefore, we follow a specific
  // order to build TaskDag for actors.
  BuildForwardTaskDagsOfType(actor_dag, TaskType::kDataTask);
  BuildForwardTaskDagsOfType(actor_dag, TaskType::kComputeTask);
  BuildForwardTaskDagsOfType(actor_dag, TaskType::kCopyTask);
  BuildForwardTaskDagsOfType(actor_dag, TaskType::kBoxingTask);
  BuildForwardTaskDagsOfType(actor_dag, TaskType::kNetTask);
}
template <typename Dtype>
void JobManager<Dtype>::BuildForwardTaskDagsOfType(
  std::shared_ptr<ActorDag<Dtype>> actor_dag, TaskType task_type) {
  DagIterator<ActorDag<Dtype>, true> dag_iterator(*actor_dag);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto actor_node =
      dynamic_cast<const OpNode<ActorMeta>*>(current_node);
    CHECK_NOTNULL(actor_node);
    auto actor_meta = actor_node->op();
    auto node_task_type = actor_meta->task_type();
    if (node_task_type != task_type) continue;  // Only care about |task_type|
    if (!actor_meta->is_forward()) continue;    // Only care about forward tasks
    int32_t task_id = actor_meta->task_id();
    auto task_dag = GetTaskDag(task_id);
    task_dag->Build();

    auto actor_name = current_node->node_name();
    if (node_task_type == TaskType::kBoxingTask) {
      task_dag->PrintDag(actor_name, true, true);
    }
  }
}
template <typename Dtype>
void JobManager<Dtype>::BuildBackwardTaskDags(
  std::shared_ptr<ActorDag<Dtype>> actor_dag) {
  DagIterator<ActorDag<Dtype>, true> dag_iterator(*actor_dag);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto actor_node =
      dynamic_cast<const OpNode<ActorMeta>*>(current_node);
    CHECK_NOTNULL(actor_node);
    auto actor_meta = actor_node->op();
    if (actor_meta->is_forward()) continue;    // Only care about backward tasks
    auto actor_name = current_node->node_name();
    int32_t task_id = actor_meta->task_id();
    auto backward_task_dag = GetTaskDag(task_id);
    backward_task_dag->Build();

    auto backward_task_type = actor_meta->task_type();
    if (backward_task_type == TaskType::kBoxingTask) {
      backward_task_dag->PrintDag(actor_name, true, true);
    }
  }
}
template <typename Dtype>
void JobManager<Dtype>::SetupTaskDags(
  std::shared_ptr<ActorDag<Dtype>> actor_dag) {
  DagIterator<ActorDag<Dtype>, true> dag_iterator(*actor_dag);
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
std::shared_ptr<TaskDag<Dtype>> JobManager<Dtype>::GetTaskDag(
  int32_t task_id) const {
  auto task_dag_it = unordered_task_id_to_dag_.find(task_id);
  CHECK(task_dag_it != unordered_task_id_to_dag_.end());
  return task_dag_it->second;
}
template <typename Dtype>
void JobManager<Dtype>::GetMemoryQuota() {
  device_memory_quota_.clear();
  host_memory_quota_.clear();
  // Currently, we assume a completely symmetric cluster: each machine has
  // exactly the same hardware spec with othrs. This enable us to infer the
  // available resources of the whole cluster more conveniently.
  // TODO(jiyuan): remove this assumption one day.
  auto& machine_descriptor
    = oneflow::TheOne<Dtype>::config_parser()->machine_descriptor();
  int32_t current_machine_id = machine_descriptor->machine_id();
  size_t current_host_memory_quota = machine_descriptor->total_host_mem();
  auto device_count = machine_descriptor->device_count();
  CHECK_GT(device_count, 0);
  // Assume all the other devices have the same spec as the first device
  auto& device_descriptor = machine_descriptor->device_descriptor(0);
  size_t device_global_mem_quota = device_descriptor->total_global_mem();

  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  for (auto& machine_thread_task : all_task_dags_) {
    auto& machine_id = machine_thread_task.first;
    auto& thread_task_dags = machine_thread_task.second;
    std::map<int32_t, size_t> device_memory_quotas;
    for (auto& thread_task : thread_task_dags) {
      auto& thread_local_id = thread_task.first;
      auto& task_dags = thread_task.second;
      if (id_map->is_device_thread(thread_local_id)) {
        size_t device_memory_quota = device_global_mem_quota;
        device_memory_quotas.insert({ thread_local_id, device_memory_quota });
      }
    }
    size_t host_memory_quota = current_host_memory_quota;
    host_memory_quota_.insert({ machine_id, host_memory_quota });
    device_memory_quota_.insert({ machine_id, device_memory_quotas });
  }
}
template <typename Dtype>
void JobManager<Dtype>::SetMemoryPolicy() {
  // TODO(jiyuan):
  // 1, estimate batch_num_each_synch_
  // 2, estimate the size of register group
  // 3, add the memory usage of PS dag
  // 4, handle with memory alignment

  // Use binary search to search an appropriate value.
  // NOTE(Chonglin): although binary search is able to find the best solution,
  // but seems O(1) method exists just by taking use of the idea I used for
  // estimating upper bound of batch size
  int32_t low_piece_size = 1;
  int32_t high_piece_size = EstimatePieceSizeUpperBound();
  // NOTE(Chonglin): set the utilize_threshold = 30 for debugging and testing
  int32_t utilize_threshold = 30;
  int32_t piece_size_each_device = 1;
  while (low_piece_size <= high_piece_size) {
    int32_t current_piece_size = low_piece_size
      + (high_piece_size - low_piece_size) / 2;
    ForwardPieceSize(current_piece_size);
    int32_t factor = EstimateMagnificationMultiple();
    float host_utilization =
      100.0 * host_memory_need_[0] / host_memory_quota_[0];
    float device_utilization
      = 100.0 * device_memory_need_[0][0] / device_memory_quota_[0][0];
    LOG(INFO) << "piece_size: " << current_piece_size;
    LOG(INFO) << "enlarging factor: " << factor;
    LOG(INFO) << "host memory utilization (%): " << host_utilization;
    LOG(INFO) << "device memory utilization (%): " << device_utilization;
    LOG(INFO) << "host memory need (bytes): " << host_memory_need_[0];
    LOG(INFO) << "device_memory_need (bytes): " << device_memory_need_[0][0];
    if (host_utilization > utilize_threshold
      || device_utilization > utilize_threshold) {
      high_piece_size = current_piece_size - 1;
    } else {
      piece_size_each_device = current_piece_size;
      low_piece_size = current_piece_size + 1;
    }
  }
  LOG(INFO) << "piece_size_each_device: " << piece_size_each_device;
}
template <typename Dtype>
void JobManager<Dtype>::ForwardPieceSize(int32_t piece_size) {
  auto& strategy_descriptor
    = oneflow::TheOne<Dtype>::config_parser()->strategy_descriptor();
  strategy_descriptor->set_piece_size_each_device(piece_size);
  SetupTaskDags(actor_dag_);
  // NOTE(Chonglin): Memory consumption of ps dag should is independent of
  // the batch size, it's only infered once
  if (!ps_actor_dag_task_setup_ && ps_actor_dag_) {
    SetupTaskDags(ps_actor_dag_);
    ps_actor_dag_task_setup_ = true;
  }
  UpdateMemoryUsage();
  GetMemoryNeed();
}
template <typename Dtype>
int32_t JobManager<Dtype>::EstimatePieceSizeUpperBound() {
  ForwardPieceSize(1);
  auto piece_1_device_mem_need_ = device_memory_need_[0][0];
  ForwardPieceSize(2);
  auto piece_2_device_mem_need_ = device_memory_need_[0][0];
  // Additional memory needed to increment piece_size by one
  auto diff = piece_2_device_mem_need_ - piece_1_device_mem_need_;

  // Take the upper bound of piece_size as the maximum number of increment
  // we can make if we make full use of the memory
  return device_memory_quota_[0][0] / diff;
}
template <typename Dtype>
void JobManager<Dtype>::ClearMemoryUsage() {
  thread_memory_usage_.clear();
  machine_memory_usage_.clear();
  MemoryUsage empty_usage;
  for (auto& machine_thread_task : all_task_dags_) {
    auto& machine_id = machine_thread_task.first;
    auto& thread_task_dags = machine_thread_task.second;
    std::map<int32_t, MemoryUsage> thread_usage;
    for (auto& thread_task : thread_task_dags) {
      auto& thread_local_id = thread_task.first;
      auto& task_dags = thread_task.second;
      thread_usage.insert({ thread_local_id, empty_usage });
    }
    thread_memory_usage_.insert({ machine_id, thread_usage });
    machine_memory_usage_.insert({ machine_id, empty_usage });
  }
}
template <typename Dtype>
void JobManager<Dtype>::UpdateMemoryUsage() {
  //ClearMemoryUsage();
  //for (auto& machine_thread_task : all_task_dags_) {
  //  auto& machine_id = machine_thread_task.first;
  //  auto& thread_task_dags = machine_thread_task.second;
  //  auto& machine_memory_usage = machine_memory_usage_[machine_id];
  //  for (auto& thread_task : thread_task_dags) {
  //    auto& thread_local_id = thread_task.first;
  //    auto& task_dags = thread_task.second;
  //    auto& thread_memory_usage
  //      = thread_memory_usage_[machine_id][thread_local_id];
  //    for (auto& task_dag : task_dags) {
  //      auto& register_infos = task_dag->register_infos();
  //      for (auto& register_info : register_infos) {
  //        AddMemoryUsage(register_info, &thread_memory_usage);
  //      }  // register_infos
  //    }  // task_dags
  //    machine_memory_usage.Add(thread_memory_usage);
  //  }  // thread_task_dags
  //}  // all_task_dags_
}
template <typename Dtype>
void JobManager<Dtype>::GetMemoryNeed() {
  host_memory_need_.clear();
  device_memory_need_.clear();
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  for (auto& machine_memory_usage : machine_memory_usage_) {
    auto machine_id = machine_memory_usage.first;
    auto memory_usage = machine_memory_usage.second;
    size_t host_memory_need = 0;
    host_memory_need += memory_usage.host_memory_size_of_data_path;
    host_memory_need += memory_usage.host_memory_size_of_model_path;
    host_memory_need_.insert({ machine_id, host_memory_need });
  }
  for (auto& thread_memory_usage_pair : thread_memory_usage_) {
    auto machine_id = thread_memory_usage_pair.first;
    auto thread_usages = thread_memory_usage_pair.second;
    std::map<int32_t, size_t> device_memory_need;
    for (auto& thread_usage_pair : thread_usages) {
      auto thread_local_id = thread_usage_pair.first;
      auto thread_memory_usage = thread_usage_pair.second;
      if (id_map->is_device_thread(thread_local_id)) {
        size_t device_memory_size = 0;
        device_memory_size
          += thread_memory_usage.device_memory_size_of_data_path;
        device_memory_size
          += thread_memory_usage.device_memory_size_of_model_path;
        device_memory_need.insert({ thread_local_id, device_memory_size });
      } else {
        CHECK_EQ(thread_memory_usage.device_memory_size_of_data_path, 0);
        CHECK_EQ(thread_memory_usage.device_memory_size_of_model_path, 0);
      }
    }
    device_memory_need_.insert({ machine_id, device_memory_need });
  }
}
template <typename Dtype>
int32_t JobManager<Dtype>::EstimateMagnificationMultiple() {
  size_t max_multiple = std::numeric_limits<size_t>::max();
  CHECK_EQ(host_memory_need_.size(), host_memory_quota_.size());
  int32_t machine_num = host_memory_need_.size();
  for (int32_t machine_id = 0; machine_id < machine_num; ++machine_id) {
    float factor
      = 1.0 * host_memory_quota_[machine_id] / host_memory_need_[machine_id];
    size_t integer_factor = std::ceil(factor);
    if (integer_factor < max_multiple) {
      max_multiple = integer_factor;
    }
  }
  CHECK_EQ(device_memory_need_.size(), device_memory_quota_.size());
  CHECK_EQ(device_memory_need_.size(), host_memory_need_.size());
  for (int32_t machine_id = 0; machine_id < machine_num; ++machine_id) {
    auto& device_memory_quota = device_memory_quota_[machine_id];
    auto& device_memory_need = device_memory_need_[machine_id];
    CHECK_EQ(device_memory_quota.size(), device_memory_need.size());
    int32_t device_num = device_memory_quota.size();
    for (int32_t device_id = 0; device_id < device_num; ++device_id) {
      float factor
        = 1.0 * device_memory_quota[device_id] / device_memory_need[device_id];
      size_t integer_factor = std::ceil(factor);
      if (integer_factor < max_multiple) {
        max_multiple = integer_factor;
      }
    }
  }
  return max_multiple;
}
template <typename Dtype>
void JobManager<Dtype>::AddMemoryUsage(const RegisterInfo& register_info,
  MemoryUsage *memory_usage) {
  //// NOTE(Chonglin): the real memory usage should take account for size
  //// of Dtype
  //if (register_info.path() == RegisterPath::kDataPath) {
  //  if (register_info.location() == RegisterLocation::kHost) {
  //    memory_usage->host_memory_size_of_data_path
  //      += register_info.memory_needed() * sizeof(Dtype);
  //  } else if (register_info.location() == RegisterLocation::kDevice) {
  //    memory_usage->device_memory_size_of_data_path
  //      += register_info.memory_needed() * sizeof(Dtype);
  //  } else {
  //    LOG(FATAL) << "Unknown RegisterLocation type";
  //  }
  //} else if (register_info.path() == RegisterPath::kModelPath) {
  //  if (register_info.location() == RegisterLocation::kHost) {
  //    memory_usage->host_memory_size_of_model_path
  //      += register_info.memory_needed() * sizeof(Dtype);
  //  } else if (register_info.location() == RegisterLocation::kDevice) {
  //    memory_usage->device_memory_size_of_model_path
  //      += register_info.memory_needed() * sizeof(Dtype);
  //  } else {
  //    LOG(FATAL) << "Unknown RegisterLocation type";
  //  }
  //} else {
  //  LOG(FATAL) << "Unknown RegisterPath type";
  //}
}
template <typename Dtype>
std::map<int32_t, std::vector<std::shared_ptr<TaskDag<Dtype>>>>
JobManager<Dtype>::GetTaskDagsOfMachine(int32_t machine_id) const {
  auto machine_it = all_task_dags_.find(machine_id);
  CHECK(machine_it != all_task_dags_.end());
  return machine_it->second;
}
INSTANTIATE_CLASS(JobManager);
}  // namespace oneflow
