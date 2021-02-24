#ifndef ONEFLOW_CORE_DEVICE_CPU_STREAM_INDEX_H_
#define ONEFLOW_CORE_DEVICE_CPU_STREAM_INDEX_H_

#include "oneflow/core/device/stream_index.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class CPUStreamIndexGenerator final : public StreamIndexGenerator {
 public:
  CPUStreamIndexGenerator();
  ~CPUStreamIndexGenerator() = default;

  stream_index_t GenerateComputeStreamIndex() override;
  stream_index_t GenerateH2DStreamIndex() override { UNIMPLEMENTED(); }
  stream_index_t GenerateD2HStreamIndex() override { UNIMPLEMENTED(); }
  stream_index_t GenerateCommNetStreamIndex();
  stream_index_t GenerateTickTockStreamIndex();
  stream_index_t GenerateIndependentTaskStreamIndex(TaskType task_type);

  bool IsComputeStreamIndex(stream_index_t index) const override;
  bool IsH2DStreamIndex(stream_index_t index) const override { UNIMPLEMENTED(); }
  bool IsD2HStreamIndex(stream_index_t index) const override { UNIMPLEMENTED(); }
  bool IsCommNetStreamIndex(stream_index_t index) const;
  bool IsTickTockStreamIndex(stream_index_t index) const;
  bool IsIndependentTaskStreamIndex(stream_index_t index) const;

 private:
  static const stream_index_t kComputeEnd = 250;
  static const stream_index_t kCommNetOffset = 1;
  static const stream_index_t kTickTockOffset = 2;
  static const stream_index_t kIndependentStart = 256;

  // for GenerateComputeStreamIndex
  stream_index_t compute_stream_num_;
  stream_index_t compute_stream_counter_;
  // for GenerateIndependentStreamIndex
  HashMap<TaskType, stream_index_t> task_type2max_stream_num_;
  HashMap<TaskType, std::vector<stream_index_t>> task_type2allocated_stream_index_vec_;
  HashMap<TaskType, size_t> task_type2allocated_stream_index_vec_index_;
  HashMap<TaskType, stream_index_t> task_type2stream_index_counter_;
};

CPUStreamIndexGenerator::CPUStreamIndexGenerator() {
  // TODO: It will not be specified by cpu_device_num in future
  compute_stream_num_ = Global<ResourceDesc, ForSession>::Get()->CpuDeviceNum();
  compute_stream_counter_ = 0;
}

stream_index_t CPUStreamIndexGenerator::GenerateComputeStreamIndex() {
  return compute_stream_counter_++ % compute_stream_num_;
}

stream_index_t CPUStreamIndexGenerator::GenerateCommNetStreamIndex() {
  return kComputeEnd + kCommNetOffset;
}

stream_index_t CPUStreamIndexGenerator::GenerateTickTockStreamIndex() {
  return kComputeEnd + kTickTockOffset;
}

stream_index_t CPUStreamIndexGenerator::GenerateIndependentTaskStreamIndex(TaskType task_type) {
  auto max_num_iter = task_type2max_stream_num_.end();
  if (IsClassRegistered<int32_t, IndependentThreadNum4TaskType>(task_type)) {
    std::unique_ptr<IndependentThreadNum4TaskType> thread_num_ptr(
        NewObj<int32_t, IndependentThreadNum4TaskType>(task_type));
    auto max_num_iter = task_type2max_stream_num_.find(task_type);
    if (max_num_iter == task_type2max_stream_num_.end()) {
      task_type2max_stream_num_.emplace(task_type, static_cast<size_t>(*thread_num_ptr));
      CHECK(task_type2allocated_stream_index_vec_.emplace(task_type, std::vector<stream_index_t>{})
                .second);
    } else {
      CHECK_EQ(max_num_iter->second, *thread_num_ptr);
      CHECK(task_type2allocated_stream_index_vec_.find(task_type)
            != task_type2allocated_stream_index_vec_.end());
    }
  }

  stream_index_t index = task_type2stream_index_counter_[task_type]++;
  if (max_num_iter != task_type2max_stream_num_.end()) {
    auto& allocated_stream_index_vec = task_type2allocated_stream_index_vec_[task_type];
    if (allocated_stream_index_vec.size() < max_num_iter->second) {
      allocated_stream_index_vec.push_back(index);
    } else {
      CHECK_EQ(allocated_stream_index_vec.size(), max_num_iter->second);
      auto iter = task_type2allocated_stream_index_vec_index_.find(task_type);
      if (iter == task_type2allocated_stream_index_vec_index_.end()) {
        iter = task_type2allocated_stream_index_vec_index_.emplace(task_type, 0).first;
      }
      CHECK_LT(iter->second, allocated_stream_index_vec.size());
      index = allocated_stream_index_vec[iter->second];
      task_type2allocated_stream_index_vec_index_[task_type] =
          (iter->second + 1) % allocated_stream_index_vec.size();
    }
  }

  return index + kIndependentStart;
}

bool CPUStreamIndexGenerator::IsComputeStreamIndex(stream_index_t index) const {
  return index < compute_stream_num_;
}

bool CPUStreamIndexGenerator::IsCommNetStreamIndex(stream_index_t index) const {
  return index == kComputeEnd + kCommNetOffset;
}

bool CPUStreamIndexGenerator::IsTickTockStreamIndex(stream_index_t index) const {
  return index == kComputeEnd + kTickTockOffset;
}

bool CPUStreamIndexGenerator::IsIndependentTaskStreamIndex(stream_index_t index) const {
  return index >= kIndependentStart;
}

REGISTER_STREAM_INDEX_GENERATOR(DeviceType::kCPU, CPUStreamIndexGenerator);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_STREAM_INDEX_H_
