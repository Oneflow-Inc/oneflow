#ifndef _NODE_MANAGER_H_
#define _NODE_MANAGER_H_
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace oneflow {
template <typename Dtype>
class JobManager;

template <typename Dtype>
class DeviceManager;

template <typename Dtype>
class Task;

template <typename Dtype>
class DeviceRegister;

template <typename Dtype>
class NodeManager {
 public:
  explicit NodeManager(std::shared_ptr<JobManager<Dtype>> job_manager);
  ~NodeManager();

  void Setup();
  // Release all memory allocated by MemoryManager
  void Release();

  // Get task by task_id
  std::shared_ptr<Task<Dtype>> GetTaskById(int32_t task_id) const;
  bool is_in_model_path(int64_t register_id) const;
  std::shared_ptr<DeviceRegister<Dtype>> GetDeviceRegister(
    int64_t register_id) const;

 private:
  int32_t machine_id_;
  std::shared_ptr<JobManager<Dtype>> job_manager_;

  // TODO(jiyuan): since |thread_local_id| is continuous, we could use a vector
  // instead of an unordered_map
  // thread_local_id -> DeviceManager of that thread
  std::unordered_map<int32_t, std::shared_ptr<DeviceManager<Dtype>>>
    device_manager_dict_;

  // task_id -> Task
  std::unordered_map<int32_t, std::shared_ptr<Task<Dtype>>> task_dict_;

  // TODO(jiyuan):
  // Does NodeManager need to manage all the DeviceRegister on this machine?
  std::unordered_map<int64_t, std::shared_ptr<DeviceRegister<Dtype>>>
    device_register_dict_;

  void CollectAllTasks();
  void InitDeviceManager();

  NodeManager(const NodeManager& other) = delete;
  NodeManager& operator=(const NodeManager& other) = delete;
};
}  // namespace oneflow
#endif  // _NODE_MANAGER_H_
