#ifndef _DAG_DAG_BUILDER_H_
#define _DAG_DAG_BUILDER_H_
#include <memory>
#include <string>
#include <map>
#include "path/path_type.h"
#include "path/path_share_policy.h"
#include "common/task_type.h"
#include "dag/segment_task_map.h"

/*
The inputs to a DagBuilder are an object of NetDescriptor which describes the 
topology of operators, and an object of StrategyDescriptor which describes the
PlacementInfo and the parallelization strategy.

Firstly, the DagBuilder builds a LogicalDag according to the NetDescriptor and 
a PlacementGroupDag according to the StrategyDescriptor. With the LogicalDag and
the PlacementGroupDag, we could generate a physical execution plan finally in 
the form of ActorDag, with SegmentDag, StageDag, PipeDag as the intermediate 
results. You could get these intermediate Dags through interfaces like
|logical_dag()|, |segment_dag()|, |stage_dag()|, |pipe_dag()|, |actor_dag()|.

Note that the ActorDag only holds the simple description of each actor (e.g., 
the name, the task_id), and the topology among actors. We create a TaskDag for
each actor to perform the real function of each actor. You could get the TaskDag
if you know its |task_id| with the function |GetTaskDag(int32_t task_id)|.
*/
namespace caffe {
class NetDescriptor;

class StrategyDescriptor;

template <typename Dtype>
class BasePath;

template <typename Dtype>
class LogicalDag;

template <typename Dtype>
class PlacementGroupDag;

template <typename Dtype>
class SegmentDag;

template <typename Dtype>
class StageDag;

template <typename Dtype>
class PipeDag;

template <typename Dtype>
class ActorDag;

template <typename Dtype>
class TaskDag;

template <typename Dtype>
class DagBuilder {
public:
  DagBuilder(const std::string& net_name,
    BasePath<Dtype>* path,
    std::shared_ptr<NetDescriptor> net_descriptor,
    std::shared_ptr<StrategyDescriptor> strategy_descriptor);
  ~DagBuilder();

  std::string net_name() const { return net_name_; }

  std::shared_ptr<TaskDag<Dtype>> GetTaskDag(int32_t task_id) const;
  std::shared_ptr<TaskDag<Dtype>> GetTaskDagByName(
    const std::string& task_name) const;

  std::shared_ptr<LogicalDag<Dtype>> logical_dag() const;
  std::shared_ptr<SegmentDag<Dtype>> segment_dag() const;
  std::shared_ptr<StageDag<Dtype>> stage_dag() const;
  std::shared_ptr<PipeDag<Dtype>> pipe_dag() const;
  std::shared_ptr<ActorDag<Dtype>> actor_dag() const;

  const SegmentTaskMap& segment_task_map() const;

  std::vector<int32_t> GetDeviceIDs(const std::string& segment_name) const;

  std::string GetTaskName(
    const std::string& segment_name, bool is_forward, int32_t device_id) const;

  // Get the TaskDag participating the cross-path dependency according to the
  // information in |sharing_detail| and |device_id|. If the TaskDag is a
  // placeholder, it will return the real TaskDag who plays the consumer or 
  // producer role in execution time.
  std::shared_ptr<TaskDag<Dtype>> GetCrossPathTaskDag(
    const PathSharingDetail& sharing_detail, int32_t device_id) const;

  void Build();
  void Setup();

  bool has_BP() const;

  DagBuilder(const DagBuilder& other) = delete;
  DagBuilder& operator=(const DagBuilder& other) = delete;
private:
  std::string net_name_;
  BasePath<Dtype>* path_;
  std::shared_ptr<NetDescriptor> net_descriptor_;
  std::shared_ptr<StrategyDescriptor> strategy_descriptor_;

  std::shared_ptr<LogicalDag<Dtype>> logical_dag_;
  std::shared_ptr<PlacementGroupDag<Dtype>> placement_group_dag_;
  std::shared_ptr<SegmentDag<Dtype>> segment_dag_;
  std::shared_ptr<StageDag<Dtype>> stage_dag_;
  std::shared_ptr<PipeDag<Dtype>> pipe_dag_;
  std::shared_ptr<ActorDag<Dtype>> actor_dag_;

  SegmentTaskMap segment_task_map_;
  void BuildSegmentTaskMap();

  // Use map instead of unordered_map to keep the order of the key
  // A collection of TaskDags on a particular thread
  using TaskDagsOfThread = std::vector<std::shared_ptr<TaskDag<Dtype>>>;
  // Map from the thread_local_id to tasks on this thread
  using TaskDagsOfMachine = std::map<int32_t, TaskDagsOfThread>;
  // Map from the machine_id to tasks on this machine
  std::map<int32_t, TaskDagsOfMachine> all_task_dags_;

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

  void BuildActorDag();
  void BuildTaskDags();
  void AddProducedRegisterInfos();
  void AddConsumedRegisterInfosInPath();

  void SetupTaskDags();

  void AddTaskDag(
    int32_t task_id, const std::string& task_name, TaskType task_type,
    bool is_forward, PathType path_type);

  void BuildForwardTaskDags();
  void BuildBackwardTaskDags();
  void BuildForwardTaskDagsOfType(TaskType expected_task_type);
};
}  // namespace caffe
#endif  // _DAG_DAG_BUILDER_H_
