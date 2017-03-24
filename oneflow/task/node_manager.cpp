#include "task/node_manager.h"
#include <vector>
#include <string>
#include <memory>
#include "common/common.h"
#include "context/one.h"
#include "task/job_manager.h"
#include "task/device_manager.h"
#include "task/device_context.h"
#include "context/machine_descriptor.h"
#include "context/config_parser.h"
#include "task/task.h"
#include "dag/task_dag.h"
#include "device/device_descriptor.h"
#include "context/resource_descriptor.h"
#include "context/solver_descriptor.h"
#include "dag/dag_iterator.h"
#include "dag/node_meta.h"
#include "task/task_context.h"
#include "task/device_register.h"
#include "task/task_op.h"

namespace caffe {
template <typename Dtype>
NodeManager<Dtype>::NodeManager(
  std::shared_ptr<JobManager<Dtype>> job_manager)
  : job_manager_(job_manager) {
  // TODO(Chonglin): machine_id_ needs to be set to it's real id
  machine_id_ = 0;
}

template <typename Dtype>
NodeManager<Dtype>::~NodeManager() {}

template <typename Dtype>
void NodeManager<Dtype>::Setup() {
  CollectAllTasks();
}

template <typename Dtype>
void NodeManager<Dtype>::Release() {
  for (auto& item : task_dict_) {
    auto& task = item.second;
    task->Release();
  }
}

template <typename Dtype>
std::shared_ptr<Task<Dtype>> NodeManager<Dtype>::GetTaskById(
  int32_t task_id) const {
  auto task_dict_it = task_dict_.find(task_id);
  CHECK(task_dict_it != task_dict_.end());
  return task_dict_it->second;
}

template <typename Dtype>
std::shared_ptr<DeviceRegister<Dtype>> NodeManager<Dtype>::GetDeviceRegister(
  int64_t register_id) const {
  auto device_register_it = device_register_dict_.find(register_id);
  CHECK(device_register_it != device_register_dict_.end());
  return device_register_it->second;
}

template <typename Dtype>
bool NodeManager<Dtype>::is_in_model_path(int64_t register_id) const {
  auto id_map = caffe::TheOne<Dtype>::id_map();
  int32_t task_id = id_map->task_id_from_register_id(register_id);
  auto task = GetTaskById(task_id);
  return task->task_dag()->path_type() == PathType::kModelUpdatePath;
}

template <typename Dtype>
void NodeManager<Dtype>::CollectAllTasks() {
  auto task_dags_of_machine = job_manager_->GetTaskDagsOfMachine(machine_id_);
  for (auto task_dags_of_device : task_dags_of_machine) {
    auto thread_local_id = task_dags_of_device.first;
    auto task_dags = task_dags_of_device.second;
    LOG(INFO) << "Collect Tasks for thread: " << thread_local_id;
    device_manager_dict_.insert({ thread_local_id,
      std::unique_ptr<DeviceManager<Dtype>>(new DeviceManager<Dtype>(
      thread_local_id)) });
    InitDeviceManager(task_dags, thread_local_id);
  }
}

template <typename Dtype>
void NodeManager<Dtype>::InitDeviceManager(
  const std::vector<std::shared_ptr<TaskDag<Dtype>>> dags,
  int32_t thread_local_id) {
  auto device_manager = device_manager_dict_[thread_local_id];
  auto device_context = device_manager->is_gpu_device()
    ? device_manager->device_context() : nullptr;
  // Allocate memory blobs for each task who hold ownerships
  for (auto&& task_dag : dags) {  // NOLINT(*)
    auto task_id = task_dag->task_id();
    task_dict_.insert({ task_id,
      std::shared_ptr<Task<Dtype>>(
      new Task<Dtype>(device_context, task_dag)) });
    auto& task = task_dict_[task_id];
    task->Setup();
    device_manager_dict_[thread_local_id]->AddTask(task_id, task);
  }
}
INSTANTIATE_CLASS(NodeManager);
}  // namespace caffe
