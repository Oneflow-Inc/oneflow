#include "context/id_map.h"
#include <utility>
#include <glog/logging.h>
#include "context/id_map.h"
#include "context/machine_descriptor.h"
#include "context/resource_descriptor.h"
#include "context/strategy_descriptor.h"
#include "context/config_parser.h"

namespace caffe {
IDMap::IDMap(std::shared_ptr<ConfigParser> config)
  : config_parser_(config) {
  auto& resource_descriptor = config_parser_->resource_descriptor();
  auto& strategy_descriptor = config_parser_->strategy_descriptor();

  machine_num_ = resource_descriptor->machine_num();
  CHECK_GT(machine_num_, 0);

  device_num_each_machine_ = resource_descriptor->device_num_per_machine();
  CHECK_GT(device_num_each_machine_, 0);
  total_device_num_ = resource_descriptor->total_device_num();

  data_thread_local_id_ = device_num_each_machine_;  // a single data thread
  boxing_thread_local_id_ = data_thread_local_id_ + 1; // a single data thread
  net_thread_local_id_ = boxing_thread_local_id_ + 1;

  int32_t logical_id = 0;
  for (int32_t machine_id = 0; machine_id < machine_num_; ++machine_id) {
    auto device_ids = resource_descriptor->machine_device_ids(machine_id);
    for (int32_t local_id = 0; local_id < device_num_each_machine_; ++local_id) {
      DeviceInfo device_info;
      device_info.logical_id = logical_id;
      device_info.machine_id = machine_id;
      device_info.local_id = local_id;
      device_info.physical_id = device_ids[local_id];
      device_info.device_id
        = device_id_from_machine_and_local(machine_id, local_id);
      devices_info_.push_back(device_info);
      device2logical_.insert({device_info.device_id, logical_id});
      ++logical_id;
    }
  }
}
IDMap::~IDMap() {
}
int32_t IDMap::machine_num() const {
  return machine_num_;
}
int32_t IDMap::device_num_each_machine() const {
  return device_num_each_machine_;
}
int32_t IDMap::total_device_num() const {
  return total_device_num_;
}

bool IDMap::is_device_thread(int32_t thread_local_id) const {
  return thread_local_id < device_num_each_machine_;
}

int64_t IDMap::data_id_from_device_and_piece(
    int32_t device_id, int64_t piece_id) const {
  int64_t data_id = static_cast<int64_t>(device_id);
  return (data_id << piece_bit_num_) | piece_id;
}

int32_t IDMap::device_id_from_data_id(int64_t data_id) const {
  return data_id >> piece_bit_num_;
}

int64_t IDMap::piece_id_from_data_id(int64_t data_id) const {
  return (data_id & (uint64_full_mask_ >> device_bit_num_));
}

int32_t IDMap::device_id_from_logical_id(int32_t logical_id) const {
  CHECK_GE(logical_id, 0);
  CHECK_LT(logical_id, total_device_num_);
  return devices_info_[logical_id].device_id;
}

int32_t IDMap::machine_id_from_logical_id(int32_t logical_id) const {
  CHECK_GE(logical_id, 0);
  CHECK_LT(logical_id, total_device_num_);
  return devices_info_[logical_id].machine_id;
}

int32_t IDMap::logical_id_from_device_id(int32_t device_id) const {
  auto it = device2logical_.find(device_id);
  CHECK(it != device2logical_.end());
  return it->second;
}

int32_t IDMap::machine_id_from_device_id(int32_t device_id) const {
  // return device_id >> thread_local_bit_num_;
  int32_t logical_id = logical_id_from_device_id(device_id);
  return devices_info_[logical_id].machine_id;
}

int32_t IDMap::local_id_from_device_id(int32_t device_id) const {
  // == 0xFF if thread_local_bit_num_ == 8
  // int32_t local_mask = uint32_full_mask_ >> (32 - thread_loal_bit_num_);
  // return (device_id & local_mask);
  int32_t logical_id = logical_id_from_device_id(device_id);
  return devices_info_[logical_id].local_id;
}

int32_t IDMap::physical_id_from_device_id(int32_t device_id) const {
  int32_t logical_id = logical_id_from_device_id(device_id);
  return devices_info_[logical_id].physical_id;
}

int32_t IDMap::device_id_from_machine_and_local(
    int32_t machine_id, int32_t local_id) const {
  CHECK_GE(local_id, 0);
  CHECK_LT(local_id, device_num_each_machine_);
  CHECK_GE(machine_id, 0);
  CHECK_LT(machine_id, machine_num_);
  return make_thread_id_with_machine_and_local(machine_id, local_id);
}

int32_t IDMap::physical_id_from_local_id(int32_t local_id)
  const {
  CHECK_GE(local_id, 0);
  CHECK_LT(local_id, device_num_each_machine_);
  int32_t machine_id = config_parser_->machine_descriptor()->machine_id();
  int32_t device_id = device_id_from_machine_and_local(machine_id, local_id);
  return physical_id_from_device_id(device_id);
}

int32_t IDMap::local_id_from_physical_id(int32_t physical_id) const {
  int32_t machine_id = config_parser_->machine_descriptor()->machine_id();
  auto& resource_descriptor = config_parser_->resource_descriptor();
  return resource_descriptor->local_from_physical(machine_id, physical_id);
}

int32_t IDMap::thread_id_from_machine_and_local(
    int32_t machine_id, int32_t local_id) const {
  if (local_id < device_num_each_machine_) {
    return device_id_from_machine_and_local(machine_id, local_id);
  } else {
    return make_thread_id_with_machine_and_local(machine_id, local_id);
  }
}

int32_t IDMap::machine_id_from_thread_id(int32_t thread_id) const {
  return thread_id >> thread_local_bit_num_;
}

int32_t IDMap::machine_id_from_task_id(int32_t task_id) const {
  return task_id >> task_local_bit_num_;
}

int32_t IDMap::local_id_from_thread_id(int32_t thread_id) const {
  // == 0xFF if thread_local_bit_num_ == 8
  int32_t local_mask = uint32_full_mask_ >> (32 - thread_local_bit_num_);
  return (thread_id & local_mask);
}

int32_t IDMap::task_local_id_from_task_id(int32_t task_id) const {
  int32_t local_task_id_mask
    = uint32_full_mask_ >> (32 - task_local_bit_num_);
  return task_id & local_task_id_mask;
}

int32_t IDMap::machine_id_from_register_id(int64_t register_id) const {
  return register_id >> (
      thread_local_bit_num_
      + task_local_bit_num_
      + register_bit_num_);
}

int32_t IDMap::thread_id_from_register_id(int64_t register_id) const {
  return register_id >> (task_local_bit_num_ + register_bit_num_);
}

int32_t IDMap::thread_local_id_from_register_id(int64_t register_id) const {
  int32_t thread_id = thread_id_from_register_id(register_id);
  return local_id_from_thread_id(thread_id);
}

int32_t IDMap::task_id_from_register_id(int64_t register_id) const {
  return register_id >> register_bit_num_;
}

int32_t IDMap::task_local_id_from_register_id(int64_t register_id) const {
  int32_t task_id = task_id_from_register_id(register_id);
  int32_t local_task_id_mask = uint32_full_mask_ >> device_bit_num_;
  return task_id & local_task_id_mask;
}

int64_t IDMap::group_id_from_register_id(int64_t register_id) const {
  return register_id >> register_local_bit_num_;
}

int32_t IDMap::group_local_id_from_register_id(int64_t register_id) const {
  int64_t group_id = group_id_from_register_id(register_id);
  int64_t group_local_id_mask = uint64_full_mask_ >> task_bit_num_;
  return group_id & group_local_id_mask;
}

int32_t IDMap::register_local_id_from_register_id(int64_t register_id) const {
  int64_t register_local_id_mask
    = uint64_full_mask_ >> (64 - register_local_bit_num_);
  return register_id & register_local_id_mask;
}

int32_t IDMap::thread_id_from_task_id(int32_t task_id) const {
  return task_id >> task_local_bit_num_;
}
//int32_t IDMap::machine_id_from_task_id(int32_t task_id) const {
//  auto thread_id = thread_id_from_task_id(task_id);
//  auto machine_id = machine_id_from_thread_id(thread_id);
//  return machine_id;
//}
int32_t IDMap::thread_local_id_from_task_id(int32_t task_id) const {
  int32_t thread_id = thread_id_from_task_id(task_id);
  return local_id_from_thread_id(thread_id);
}

int32_t IDMap::device_thread_local_id(int32_t local_id) const {
  CHECK_GE(local_id, 0);
  CHECK_LT(local_id, device_num_each_machine_);
  return local_id;
}

int32_t IDMap::data_thread_local_id() const {
  return data_thread_local_id_;
}

int32_t IDMap::boxing_thread_local_id() const {
  return boxing_thread_local_id_;
}

int32_t IDMap::net_thread_local_id() const {
  return net_thread_local_id_;
}

int32_t IDMap::make_thread_id_with_machine_and_local(int32_t machine_id,
  int32_t local_id) const {
  return (machine_id << thread_local_bit_num_) | local_id;
}

int32_t IDMap::task_id_from_thread_id_and_task_local_id(
  int32_t thread_id,
  int32_t task_local_id) const {
  return (thread_id << task_local_bit_num_) | task_local_id;
}

int64_t IDMap::group_id_from_task_id_and_group_local_id(int32_t task_id,
  int32_t group_local_id) const {
  int64_t group_id = task_id;
  return (group_id << group_local_bit_num_) | group_local_id;
}

int64_t IDMap::register_id_from_group_id_and_register_local_id(int64_t group_id,
  int32_t register_local_id) const {
  return (group_id << register_local_bit_num_) | register_local_bit_num_;
}

int32_t IDMap::task_id_from_group_id(int64_t group_id) const {
  return group_id >> group_local_bit_num_;
}

int32_t IDMap::new_task_local_id(int32_t thread_id) {
  auto task_it = thread_id_to_task_local_id_counter_.find(thread_id);
  int32_t task_local_id = -1;
  if (task_it == thread_id_to_task_local_id_counter_.end()) {
    thread_id_to_task_local_id_counter_.insert({ thread_id, 0 });
    task_local_id = thread_id_to_task_local_id_counter_[thread_id];
  } else {
    task_local_id = ++(thread_id_to_task_local_id_counter_[thread_id]);
  }
  return task_local_id;
}

int32_t IDMap::new_group_local_id(int32_t task_id) {
  auto group_it = task_id_to_group_local_id_counter_.find(task_id);
  int32_t group_local_id = -1;
  if (group_it == task_id_to_group_local_id_counter_.end()) {
    task_id_to_group_local_id_counter_.insert({ task_id, 0 });
    group_local_id = task_id_to_group_local_id_counter_[task_id];
  } else {
    group_local_id = ++(task_id_to_group_local_id_counter_[task_id]);
  }
  return group_local_id;
}

int32_t IDMap::new_register_local_id(int64_t group_id) {
  auto register_it = group_id_to_register_local_id_counter_.find(group_id);
  int32_t register_local_id = -1;
  if (register_it == group_id_to_register_local_id_counter_.end()) {
    group_id_to_register_local_id_counter_.insert({group_id, 0 });
    register_local_id = group_id_to_register_local_id_counter_[group_id];
  } else {
    register_local_id = ++(group_id_to_register_local_id_counter_[group_id]);
  }
  return register_local_id;
}

}  // namespace caffe
