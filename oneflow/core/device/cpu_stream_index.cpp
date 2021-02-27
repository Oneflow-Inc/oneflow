/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/device/cpu_stream_index.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

CPUStreamIndexGenerator::CPUStreamIndexGenerator()
    : next_stream_index_(0), compute_stream_index_counter_(0) {
  compute_stream_index_begin_ = next_stream_index_;
  // TODO: It will not be specified by cpu_device_num in future
  compute_stream_num_ = Global<ResourceDesc, ForSession>::Get()->CpuDeviceNum();
  next_stream_index_ += compute_stream_num_;
  comm_net_stream_index_ = next_stream_index_;
  next_stream_index_++;
  tick_tock_stream_index_ = next_stream_index_;
  next_stream_index_++;
}

stream_index_t CPUStreamIndexGenerator::GenerateComputeStreamIndex() {
  return compute_stream_index_begin_ + (compute_stream_index_counter_++ % compute_stream_num_);
}

stream_index_t CPUStreamIndexGenerator::GenerateCommNetStreamIndex() {
  return comm_net_stream_index_;
}

stream_index_t CPUStreamIndexGenerator::GenerateTickTockStreamIndex() {
  return tick_tock_stream_index_;
}

stream_index_t CPUStreamIndexGenerator::GenerateIndependentTaskStreamIndex(TaskType task_type) {
  auto max_num_iter = task_type2max_stream_num_.end();
  if (IsClassRegistered<int32_t, IndependentThreadNum4TaskType>(task_type)) {
    std::unique_ptr<IndependentThreadNum4TaskType> thread_num_ptr(
        NewObj<int32_t, IndependentThreadNum4TaskType>(task_type));
    const size_t max_num = static_cast<size_t>(*thread_num_ptr.get());
    auto max_num_iter = task_type2max_stream_num_.find(task_type);
    if (max_num_iter == task_type2max_stream_num_.end()) {
      task_type2max_stream_num_.emplace(task_type, max_num);
      CHECK(task_type2allocated_stream_index_vec_.emplace(task_type, std::vector<stream_index_t>{})
                .second);
    } else {
      CHECK_EQ(max_num_iter->second, max_num);
      CHECK(task_type2allocated_stream_index_vec_.find(task_type)
            != task_type2allocated_stream_index_vec_.end());
    }
  }

  stream_index_t index = next_stream_index_;
  if (max_num_iter != task_type2max_stream_num_.end()) {
    auto& allocated_stream_index_vec = task_type2allocated_stream_index_vec_[task_type];
    if (allocated_stream_index_vec.size() < max_num_iter->second) {
      allocated_stream_index_vec.push_back(index);
      next_stream_index_++;
    } else {
      CHECK_EQ(allocated_stream_index_vec.size(), max_num_iter->second);
      auto& next = task_type2allocated_stream_index_vec_index_[task_type];
      index = allocated_stream_index_vec[next++];
      next %= allocated_stream_index_vec.size();
    }
  } else {
    next_stream_index_++;
  }
  return index;
}

REGISTER_STREAM_INDEX_GENERATOR(DeviceType::kCPU, CPUStreamIndexGenerator);

}  // namespace oneflow
