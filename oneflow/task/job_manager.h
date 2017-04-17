#ifndef _JOB_MANAGER_H_
#define _JOB_MANAGER_H_
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <map>
#include <unordered_map>
#include "common/task_type.h"
#include "dag/task_dag.h"
#include "context/config_parser.h"
#include "dag/dag_builder.h"
#include "task/memory_usage.h"
#include "dag/register_info.h"
#include "path/base_path.h"

/*
We have several levels of managers to handle with the decomposed tasks:
(1) JobManager may act as a centralized dictionary of resources at job-level, 
including the TaskDags of all the nodes (machines).
(2) NodeManager handles with the tasks on current node (machine), TODO(jiyuan):
change name NodeManager to NodeManager.
(3) DeviceManager handles with the tasks on a particular device/thread of this
machine.
*/
namespace oneflow {
template <typename Dtype>
class ActorDag;
template <typename Dtype>
class TaskDag;
template <typename Dtype>
class DagBuilder;

template <typename Dtype>
class JobManager {
 public:
  JobManager(std::shared_ptr<ActorDag<Dtype>> actor_dag,
    std::shared_ptr<ActorDag<Dtype>> ps_actor_dag);
  ~JobManager();

  void Init();
  std::shared_ptr<TaskDag<Dtype>> GetTaskDag(int32_t task_id) const;
  std::map<int32_t, std::vector<std::shared_ptr<TaskDag<Dtype>>>>
    GetTaskDagsOfMachine(int32_t machine_id) const;

 private:
  std::shared_ptr<ActorDag<Dtype>> actor_dag_;
  std::shared_ptr<ActorDag<Dtype>> ps_actor_dag_;

  // Whether PS dag tasks have already been setup, since it's independent of
  // batch_size, it needs to be setup only once
  bool ps_actor_dag_task_setup_;

  void ProcessAllTaskDags();
  void CollectTaskDags(std::shared_ptr<ActorDag<Dtype>> actor_dag);
  void AddTaskDag(int32_t task_id, const std::string& task_name,
    TaskType task_type, bool is_forward, bool is_ps_dag, PathType path_type);

  void BuildTaskDags(std::shared_ptr<ActorDag<Dtype>> actor_dag);
  void BuildForwardTaskDags(std::shared_ptr<ActorDag<Dtype>> actor_dag);
  void BuildForwardTaskDagsOfType(
    std::shared_ptr<ActorDag<Dtype>> actor_dag, TaskType task_type);
  void BuildBackwardTaskDags(
    std::shared_ptr<ActorDag<Dtype>> actor_dag);

  void ClearMemoryUsage();
  void UpdateMemoryUsage();
  void AddMemoryUsage(
    const RegisterInfo& register_info, MemoryUsage *memory_usage);
  void SetupTaskDags(std::shared_ptr<ActorDag<Dtype>> actor_dag);
  void GetMemoryQuota();
  void SetMemoryPolicy();
  void GetMemoryNeed();

  // Estimate how many multiples we can enlarge the batch_size_each_device
  int32_t EstimateMagnificationMultiple();
  void ForwardPieceSize(int32_t piece_size);
  int32_t EstimatePieceSizeUpperBound();

  // <task_id, task_dag>
  // We use two dictionaries, one with keeping the order of task_id,
  // the other without keeping order of task_id. See id_map.h, using int32_t is
  // sufficient to hold the task_id
  std::map<int32_t, std::shared_ptr<TaskDag<Dtype>>>
    ordered_task_id_to_dag_;
  std::unordered_map<int32_t, std::shared_ptr<TaskDag<Dtype>>>
    unordered_task_id_to_dag_;
  // <task_name, task_dag>
  std::unordered_map<std::string, std::shared_ptr<TaskDag<Dtype>>>
    task_name_to_dag_;

  // NOTE(jiyuan): we use |MemoryUsage| to indicate the detailed memory needs of
  // each thread/machine; use |MemoryQuota| to indicate the maximum amount of
  // the available memory on a device or a machine; use |MemoryNeed| to indicate
  // the total memory needed of each device/machine.

  // The detailed memory usage of each thread
  std::map<int32_t, std::map<int32_t, MemoryUsage>> thread_memory_usage_;
  // The detailed memory usage of each machine
  std::map<int32_t, MemoryUsage> machine_memory_usage_;

  // Use map instead of unordered_map to keep the order of the key
  // A collection of TaskDags on a particular thread
  using TaskDagsOfThread = std::vector<std::shared_ptr<TaskDag<Dtype>>>;
  // Map from the thread_local_id to tasks on this thread
  using TaskDagsOfMachine = std::map<int32_t, TaskDagsOfThread>;
  // Map from the machine_id to tasks on this machine
  std::map<int32_t, TaskDagsOfMachine> all_task_dags_;

  // The host memory quota of each machine, <machine_id, quota>
  std::map<int32_t, size_t> host_memory_quota_;
  // The memory quota of each device, <machine_id, <device_local_id, quota>>
  std::map<int32_t, std::map<int32_t, size_t>> device_memory_quota_;

  // The host memory need of each machine, <machine_id, need>
  std::map<int32_t, size_t> host_memory_need_;
  // The device memory need of each device,
  // <machine_id, <device_local_id, need>>
  std::map<int32_t, std::map<int32_t, size_t>> device_memory_need_;

  JobManager(const JobManager& other) = delete;
  JobManager& operator=(const JobManager& other) = delete;
};
}  // namespace oneflow
#endif  // _JOB_MANAGER_H_
