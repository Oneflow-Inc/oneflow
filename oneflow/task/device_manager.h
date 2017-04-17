#ifndef _DEVICE_DEVICE_MANAGER_H_
#define _DEVICE_DEVICE_MANAGER_H_
#include <memory>
#include <unordered_map>
#include <fstream>
#include <glog/logging.h>

/*
Contains all the tasks bound to the particular device/thread.
*/
namespace oneflow {
class DeviceDescriptor;

template <typename Dtype>
class Task;

template <typename Dtype>
class DeviceContext;

template <typename Dtype>
class DeviceManager {
 public:
   explicit DeviceManager(int32_t thread_local_id);
  ~DeviceManager();

  void Setup();
  void AddTask(int32_t task_id, std::shared_ptr<Task<Dtype>> task);
  std::shared_ptr<Task<Dtype>> GetTask(int32_t task_id) const;

  bool is_gpu_device() const { return is_gpu_; }
  std::shared_ptr<DeviceContext<Dtype>> device_context() const;

 private:
  int32_t thread_local_id_;
  bool is_gpu_{ false };
  std::shared_ptr<DeviceContext<Dtype>> device_context_;  // For GPU device

  // task_id -> task
  std::unordered_map<int32_t, std::shared_ptr<Task<Dtype>>> task_dict_;

  DeviceManager(const DeviceManager& other) = delete;
  DeviceManager& operator=(const DeviceManager& other) = delete;
};
}  // namespace oneflow
#endif  // _DEVICE_DEVICE_MANAGER_H_
