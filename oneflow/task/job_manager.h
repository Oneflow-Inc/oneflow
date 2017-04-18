#ifndef _JOB_MANAGER_H_
#define _JOB_MANAGER_H_
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <map>
#include <unordered_map>
#include "common/task_type.h"
//#include "dag/task_dag.h"
#include "context/config_parser.h"
//#include "dag/dag_builder.h"
#include "task/memory_usage.h"
//#include "dag/register_info.h"

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
class JobManager {
 public:
  JobManager();
  ~JobManager();

  void Init();

 private:
  // The detailed memory usage of each thread
  std::map<int32_t, std::map<int32_t, MemoryUsage>> thread_memory_usage_;
  // The detailed memory usage of each machine
  std::map<int32_t, MemoryUsage> machine_memory_usage_;

  // Use map instead of unordered_map to keep the order of the key
  // A collection of TaskDags on a particular thread
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
