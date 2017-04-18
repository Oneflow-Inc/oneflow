#include "task/task_context.h"
#include <glog/logging.h>
#include <string>
#include <vector>
#include <algorithm>
#include "common/common.h"
#include "common/task_type.h"
#include "device/device_alternate.h"
#include "memory/blob.h"
#include "task/device_register.h"
#include "common/str_util.h"
#include "context/one.h"
#include "context/resource_descriptor.h"
#include "task/node_manager.h"
#include "task/task.h"
#include "task/task_op.h"

namespace oneflow {

template <typename Dtype>
TaskContext<Dtype>::TaskContext(Task<Dtype>* task)
  : task_(task), gpu_chunk_(DeviceType::kGPU), cpu_chunk_(DeviceType::kCPU),
  cpu_pinned_chunk_(DeviceType::kCPUPinned){
}

template <typename Dtype>
TaskContext<Dtype>::~TaskContext() {}

template <typename Dtype>
void TaskContext<Dtype>::Setup() {
  InitFromTaskDag();
  AllocateMemoryChunk();
  CreateDeviceRegisters();
}

template <typename Dtype>
void TaskContext<Dtype>::InitFromTaskDag() {
}

template <typename Dtype>
void TaskContext<Dtype>::AllocateMemoryChunk() {
}

template <typename Dtype>
void TaskContext<Dtype>::CreateDeviceRegisters() {
}

template <typename Dtype>
const std::vector<int64_t>& TaskContext<Dtype>::register_group_ids() const {
  return register_group_ids_;
}

template <typename Dtype>
const std::vector<int64_t>& TaskContext<Dtype>::register_ids_in_group(
  int64_t group_id) const {
  auto register_ids_it = group_id_to_register_ids_.find(group_id);
  CHECK(register_ids_it != group_id_to_register_ids_.end());
  return register_ids_it->second;
}

template <typename Dtype>
TaskContext<Dtype>::MemoryChunk::MemoryChunk(DeviceType device_type)
  : memory_needed_(0), memory_offset_(0), device_type_(device_type),
  device_id_(-2) {
}

template <typename Dtype>
TaskContext<Dtype>::MemoryChunk::~MemoryChunk() {
  if (memory_needed_ > 0) {
    MemoryManager::Get()->Free(handle_);
  }
}

template <typename Dtype>
void TaskContext<Dtype>::MemoryChunk::SetDeviceId(int32_t device_id) {
  device_id_ = device_id;
}

template <typename Dtype>
void TaskContext<Dtype>::MemoryChunk::SetMemoryNeeded(int64_t memory_needed) {
  memory_needed_ = memory_needed;
  CHECK(device_type_ != DeviceType::kUnknown);
  CHECK_GT(memory_needed_, 0);
  CHECK(device_id_ != -2);

  MemoryManager::Context ctx;
  ctx.dev_type = device_type_;
  ctx.dev_id = device_id_; // Just set dev_id to -1 as it's on CPU
  handle_ = MemoryManager::Get()->Alloc(memory_needed_, ctx);
  data_ptr_ = static_cast<char*>(handle_.dptr);
}

template <typename Dtype>
void TaskContext<Dtype>::MemoryChunk::ConsumeMemory(int64_t block_size) {
  memory_offset_ += block_size;
  CHECK_LT(memory_offset_, memory_needed_);
  data_ptr_ = static_cast<char*>(handle_.dptr) + memory_offset_;
}

template <typename Dtype>
char* TaskContext<Dtype>::MemoryChunk::data_ptr() const {
  return data_ptr_;
}

INSTANTIATE_CLASS(TaskContext);
}  // namespace oneflow
