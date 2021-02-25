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
#ifndef ONEFLOW_CORE_DEVICE_CPU_STREAM_INDEX_H_
#define ONEFLOW_CORE_DEVICE_CPU_STREAM_INDEX_H_

#include "oneflow/core/device/stream_index.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class CPUStreamIndexGenerator final : public StreamIndexGenerator {
 public:
  CPUStreamIndexGenerator();
  OF_DISALLOW_COPY_AND_MOVE(CPUStreamIndexGenerator);
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

 private:
  stream_index_t next_stream_index_;
  stream_index_t compute_stream_index_begin_;
  stream_index_t compute_stream_num_;
  stream_index_t comm_net_stream_index_;
  stream_index_t tick_tock_stream_index_;
  // for GenerateComputeStreamIndex
  stream_index_t compute_stream_index_counter_;
  // for GenerateIndependentStreamIndex
  HashMap<TaskType, size_t> task_type2max_stream_num_;
  HashMap<TaskType, std::vector<stream_index_t>> task_type2allocated_stream_index_vec_;
  HashMap<TaskType, size_t> task_type2allocated_stream_index_vec_index_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_STREAM_INDEX_H_
