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
#include "device/device_descriptor.h"
#include "context/resource_descriptor.h"
#include "context/solver_descriptor.h"
#include "task/task_context.h"
#include "task/device_register.h"
#include "task/task_op.h"

namespace oneflow {
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
}

template <typename Dtype>
void NodeManager<Dtype>::CollectAllTasks() {
}

template <typename Dtype>
void NodeManager<Dtype>::InitDeviceManager() {
}
INSTANTIATE_CLASS(NodeManager);
}  // namespace oneflow
