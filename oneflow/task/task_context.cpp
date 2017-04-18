#include "task/task_context.h"
#include <glog/logging.h>
#include <string>
#include <vector>
#include <algorithm>
//#include "common/common.h"
//#include "common/task_type.h"
//#include "dag/task_dag.h"
//#include "device/device_alternate.h"
//#include "memory/blob.h"
//#include "layers/base_layer.h"
//#include "task/device_register.h"
//#include "common/str_util.h"
//#include "context/one.h"
//#include "context/resource_descriptor.h"
//#include "task/node_manager.h"
#include "task/task.h"
// #include "task/task_op.h"

namespace oneflow {

TaskContext::TaskContext(Task* task)
  : task_(task)
  // , gpu_chunk_(DeviceType::kGPU), cpu_chunk_(DeviceType::kCPU),
  // cpu_pinned_chunk_(DeviceType::kCPUPinned)
{}

TaskContext::~TaskContext() {}

void TaskContext::Setup() {
  //InitFromTaskDag();
  //AllocateMemoryChunk();
  //CreateDeviceRegisters();
}

//template <typename Dtype>
//void TaskContext<Dtype>::InitFromTaskDag() {
//  auto id_map = caffe::TheOne<Dtype>::id_map();
//  register_group_ids_ = task_->task_dag()->GetProducedGroupIds();
//  for (auto group_id : register_group_ids_) {
//    auto& register_info = task_->task_dag()->GetProducedRegisterInfo(group_id);
//    group_id_to_register_info_.insert({ group_id, register_info});
//    int32_t group_size = task_->task_dag()->GetProducedGroupSize(group_id);
//    for (int32_t i = 0; i < group_size; ++i) {
//      int64_t register_id
//        = id_map->register_id_from_group_id_and_register_local_id(group_id, i);
//      if (i == 0) {
//        std::vector<int64_t> register_ids;
//        register_ids.push_back(register_id);
//        group_id_to_register_ids_.insert({ group_id, register_ids });
//      } else {
//        group_id_to_register_ids_[group_id].push_back(register_id);
//      }
//    }
//  }
//}
//
//template <typename Dtype>
//void TaskContext<Dtype>::AllocateMemoryChunk() {
//  int64_t gpu_memory_needed = 0;
//  int64_t cpu_memory_needed = 0;
//  int64_t cpu_pinned_memory_needed = 0;
//  for (auto group_id : register_group_ids_) {
//    auto& register_info = group_id_to_register_info_[group_id];
//    DeviceType device_type = register_info.device_type();
//    int64_t register_element_num = register_info.total_element_num();
//    int32_t group_size = group_id_to_register_ids_[group_id].size();
//    int64_t group_element_num = register_element_num * group_size;
//    if (device_type == DeviceType::kGPU) {
//      gpu_memory_needed += group_element_num * sizeof(Dtype);
//    } else if (device_type == DeviceType::kCPU) {
//      cpu_memory_needed += group_element_num * sizeof(Dtype);
//    } else if (device_type == DeviceType::kCPUPinned) {
//      cpu_pinned_memory_needed += group_element_num * sizeof(Dtype);
//    }
//  }
//  if (gpu_memory_needed > 0) {
//    auto& id_map = caffe::TheOne<Dtype>::id_map();
//    int32_t thread_id = id_map->thread_id_from_task_id(task_->task_id());
//    int32_t physical_id = id_map->physical_id_from_device_id(thread_id);
//    gpu_chunk_.SetDeviceId(physical_id);
//    gpu_chunk_.SetMemoryNeeded(gpu_memory_needed);
//  }
//  if (cpu_memory_needed > 0) {
//    gpu_chunk_.SetDeviceId(-1);
//    gpu_chunk_.SetMemoryNeeded(cpu_memory_needed);
//  }
//  if (cpu_pinned_memory_needed > 0) {
//    gpu_chunk_.SetDeviceId(-1);
//    gpu_chunk_.SetMemoryNeeded(cpu_pinned_memory_needed);
//  }
//}
//
//template <typename Dtype>
//void TaskContext<Dtype>::CreateDeviceRegisters() {
//  for (auto group_id : register_group_ids_) {
//    auto& register_info = group_id_to_register_info_[group_id];
//    auto& register_ids = group_id_to_register_ids_[group_id];
//    for (auto register_id : register_ids) {
//      std::shared_ptr<DeviceRegister<Dtype>> device_register;
//      void *data_ptr = nullptr;
//      if (register_info.device_type() == DeviceType::kCPU) {
//        data_ptr = cpu_chunk_.data_ptr();
//        cpu_chunk_.ConsumeMemory(
//          register_info.total_element_num() * sizeof(Dtype));
//      } else if (register_info.device_type() == DeviceType::kCPUPinned) {
//        data_ptr = cpu_pinned_chunk_.data_ptr();
//        cpu_pinned_chunk_.ConsumeMemory(
//          register_info.total_element_num() * sizeof(Dtype));
//      } else if (register_info.device_type() == DeviceType::kGPU) {
//        data_ptr = gpu_chunk_.data_ptr();
//        gpu_chunk_.ConsumeMemory(
//          register_info.total_element_num() * sizeof(Dtype));
//      }
//      device_register.reset(new DeviceRegister<Dtype>(data_ptr, register_info));
//      device_register_dict_.insert({ register_id, device_register });
//    }
//  }
//}

const std::vector<int64_t>& TaskContext::register_group_ids() const {
  return register_group_ids_;
}

const std::vector<int64_t>& TaskContext::register_ids_in_group(
  int64_t group_id) const {
  auto register_ids_it = group_id_to_register_ids_.find(group_id);
  CHECK(register_ids_it != group_id_to_register_ids_.end());
  return register_ids_it->second;
}

//TaskContext::MemoryChunk::MemoryChunk(DeviceType device_type)
//  : memory_needed_(0), memory_offset_(0), device_type_(device_type),
//  device_id_(-2) {
//}
//
//template <typename Dtype>
//TaskContext<Dtype>::MemoryChunk::~MemoryChunk() {
//  if (memory_needed_ > 0) {
//    MemoryManager::Get()->Free(handle_);
//  }
//}
//
//template <typename Dtype>
//void TaskContext<Dtype>::MemoryChunk::SetDeviceId(int32_t device_id) {
//  device_id_ = device_id;
//}
//
//template <typename Dtype>
//void TaskContext<Dtype>::MemoryChunk::SetMemoryNeeded(int64_t memory_needed) {
//  memory_needed_ = memory_needed;
//  CHECK(device_type_ != DeviceType::kUnknown);
//  CHECK_GT(memory_needed_, 0);
//  CHECK(device_id_ != -2);
//
//  MemoryManager::Context ctx;
//  ctx.dev_type = device_type_;
//  ctx.dev_id = device_id_; // Just set dev_id to -1 as it's on CPU
//  handle_ = MemoryManager::Get()->Alloc(memory_needed_, ctx);
//  data_ptr_ = static_cast<char*>(handle_.dptr);
//}
//
//template <typename Dtype>
//void TaskContext<Dtype>::MemoryChunk::ConsumeMemory(int64_t block_size) {
//  memory_offset_ += block_size;
//  CHECK_LT(memory_offset_, memory_needed_);
//  data_ptr_ = static_cast<char*>(handle_.dptr) + memory_offset_;
//}
//
//template <typename Dtype>
//char* TaskContext<Dtype>::MemoryChunk::data_ptr() const {
//  return data_ptr_;
//}

}  // namespace oneflow
