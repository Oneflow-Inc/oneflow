#ifndef _TASK_CONTEXT_H_
#define _TASK_CONTEXT_H_
#include <memory>
#include <vector>
#include <queue>
#include <string>
#include "dag/dag_node.h"
#include "layers/base_layer.h"
#include "memory/memory_manager.h"
#include "dag/register_info.h"

/*
Each task needs some resources to perform the operation, such as memory, cuda
related handles, cuda streams.

TaskContext managers all the required memory resources. The device-specific
resources like cuda handle, streams are managed by DeviceContext.

Each task may generate several types of DeviceRegister. We name each type as a
group. To support pipelining, we further construct a set of DeviceRegisters for
a particular group.

For each register, we maintain the reference to its handle at several places:
(1) TaskContext, which owns a task-specific registers
(2) DeviceManager, which owns a device-specific registers
(3) NodeManager, which owns a node-specific registers
*/

namespace caffe {
template <typename Dtype>
class DeviceRegister;

template <typename Dtype>
class Task;

template <typename Dtype>
class TaskContext {
 public:
   explicit TaskContext(Task<Dtype>* task);
  ~TaskContext();

  void Setup();
  const std::vector<int64_t>& register_group_ids() const;
  const std::vector<int64_t>& register_ids_in_group(int64_t group_id) const;
  char* cpu_chunk_data_ptr() const {
    return (&cpu_chunk_)->data_ptr();
  }

private:
  class MemoryChunk {
  public:
    explicit MemoryChunk(DeviceType device_type);
    ~MemoryChunk();

    void SetDeviceId(int32_t device_id);
    void SetMemoryNeeded(int64_t memory_needed);
    void ConsumeMemory(int64_t block_size);
    char* data_ptr() const;

  private:
    DeviceType device_type_;

    // -2: uninitialized
    // -1: on CPU
    // Non-negative value: on GPU
    int32_t device_id_;
    MemoryManager::Handle handle_;
    int64_t memory_needed_;
    int64_t memory_offset_;
    char* data_ptr_;

    MemoryChunk(const MemoryChunk& other) = delete;
    MemoryChunk& operator=(const MemoryChunk& other) = delete;
  };

 private:
  using RegisterDict
    = std::unordered_map <int64_t, std::shared_ptr<DeviceRegister<Dtype>>>;

 private:
  Task<Dtype>* task_;
  RegisterDict device_register_dict_;

  std::vector<int64_t> register_group_ids_;
  std::unordered_map<int64_t, std::vector<int64_t>> group_id_to_register_ids_;
  std::unordered_map<int64_t, RegisterInfo> group_id_to_register_info_;

  MemoryChunk gpu_chunk_;
  MemoryChunk cpu_chunk_;
  MemoryChunk cpu_pinned_chunk_;

  // Initialize |register_group_ids_|, |group_id_to_register_ids_|,
  // |group_id_to_register_infos_|.
  void InitFromTaskDag();
  void AllocateMemoryChunk();
  void CreateDeviceRegisters();

  TaskContext(const TaskContext& other) = delete;
  TaskContext& operator=(const TaskContext& other) = delete;
};

}  // namespace caffe
#endif  // _TASK_CONTEXT_H_
