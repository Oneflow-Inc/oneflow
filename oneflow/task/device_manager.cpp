#include "task/device_manager.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <fstream>
#include "context/one.h"
#include "context/config_parser.h"
#include "context/machine_descriptor.h"
#include "context/resource_descriptor.h"
#include "context/id_map.h"
#include "common/common.h"
#include "device/device_descriptor.h"
#include "task/device_context.h"
#include "task/task.h"

namespace caffe {
template <typename Dtype>
DeviceManager<Dtype>::DeviceManager(int32_t thread_local_id)
  : thread_local_id_(thread_local_id), device_context_(nullptr) {
  auto& id_map = caffe::TheOne<Dtype>::id_map();
  auto machine_descriptor
    = caffe::TheOne<Dtype>::config_parser()->machine_descriptor();
  //auto device_num_per_machine = resource_descriptor->device_num_per_machine();
  is_gpu_ = id_map->is_device_thread(thread_local_id_);
  if (is_gpu_) {
    int32_t physical_id = id_map->physical_id_from_local_id(thread_local_id);
    device_context_.reset(new DeviceContext<Dtype>(physical_id));
  }
}

template <typename Dtype>
DeviceManager<Dtype>::~DeviceManager() {}

template <typename Dtype>
void DeviceManager<Dtype>::AddTask(int32_t task_id,
  std::shared_ptr<Task<Dtype>> task) {
  CHECK_EQ(task_dict_.count(task_id), 0);
  task_dict_.insert({task_id, task});
}

template <typename Dtype>
std::shared_ptr<Task<Dtype>> DeviceManager<Dtype>::GetTask(
  int32_t task_id) const {
  auto task_it = task_dict_.find(task_id);
  CHECK(task_it != task_dict_.end());
  return task_it->second;
}

template <typename Dtype>
void DeviceManager<Dtype>::Setup() {
  for (auto& it : task_dict_) {
    it.second->Setup();
  }
}

template <typename Dtype>
std::shared_ptr<DeviceContext<Dtype>>
DeviceManager<Dtype>::device_context() const {
  CHECK(is_gpu_);
  return device_context_;
}

INSTANTIATE_CLASS(DeviceManager);
}  // namespace caffe
