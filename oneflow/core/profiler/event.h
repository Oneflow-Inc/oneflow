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
#ifndef ONEFLOW_CORE_PROFILER_EVENT_H_
#define ONEFLOW_CORE_PROFILER_EVENT_H_

#include "nlohmann/json.hpp"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace profiler {

enum class EventType { kCustom, kKernel };

class IEvent {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IEvent);

  IEvent() = delete;
  explicit IEvent(const std::string& name) : name_(name) {}

  virtual std::string Key() = 0;
  virtual nlohmann::json ToJson();
  virtual ~IEvent() = default;
  virtual void Start();
  virtual void Finish();

  const std::string& GetName() const;
  time_t GetDuration();

 protected:
  std::string name_;
  time_t started_at_ = 0;
  time_t finished_at_ = 0;
};

class CustomEvent final : public IEvent {
 public:
  std::string Key() override;

  nlohmann::json ToJson() override;

  static std::shared_ptr<CustomEvent> Create(const std::string& name);

 private:
  explicit CustomEvent(const std::string& custom_name) : IEvent(custom_name) {}
};

#if defined(WITH_CUDA)

class CUDAEventPair {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CUDAEventPair);

  explicit CUDAEventPair(cudaStream_t cuda_stream) : cuda_stream_(cuda_stream) {
    OF_CUDA_CHECK(cudaEventCreate(&cuda_event_start_));
    OF_CUDA_CHECK(cudaEventCreate(&cuda_event_finish_));
  }

  void Start() { OF_CUDA_CHECK(cudaEventRecord(cuda_event_start_, cuda_stream_)); }

  void Finish() { OF_CUDA_CHECK(cudaEventRecord(cuda_event_finish_, cuda_stream_)); }

  double ElapsedTime() const {
    float elapsed_time_ms = 0;
    OF_CUDA_CHECK(cudaEventSynchronize(cuda_event_start_));
    OF_CUDA_CHECK(cudaEventSynchronize(cuda_event_finish_));
    OF_CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_ms, cuda_event_start_, cuda_event_finish_));
    return elapsed_time_ms * 1000.0;  // convert to us
  }

  ~CUDAEventPair() {
    if (cuda_event_start_) { OF_CUDA_CHECK(cudaEventDestroy(cuda_event_start_)); }
    if (cuda_event_finish_) { OF_CUDA_CHECK(cudaEventDestroy(cuda_event_finish_)); }
  }

 private:
  cudaStream_t cuda_stream_ = nullptr;
  cudaEvent_t cuda_event_start_ = nullptr;
  cudaEvent_t cuda_event_finish_ = nullptr;
};

#endif  // WITH_CUDA

class KernelEvent final : public IEvent {
 public:
  std::string Key() override;

  nlohmann::json ToJson() override;

  static std::shared_ptr<KernelEvent> Create(
      const std::string& name, const std::function<std::vector<Shape>(void)>& shape_getter);

  void RecordShape(const Shape& shape);

  void Start() override;
  void Finish() override;

#if defined(WITH_CUDA)
  void InitCudaEventPair(cudaStream_t cuda_stream) {
    cuda_event_pair_ = std::make_shared<CUDAEventPair>(cuda_stream);
  }

  void SetMemorySize(int64_t memory_size) { memory_size_ = memory_size; }
#endif  // WITH_CUDA

 private:
  explicit KernelEvent(const std::string& kernel_name,
                       const std::function<std::vector<Shape>(void)>& shape_getter)
      : IEvent(kernel_name) {
    if (shape_getter) { input_shapes_ = shape_getter(); }
  }

#if defined(WITH_CUDA)
  std::shared_ptr<CUDAEventPair> cuda_event_pair_ = nullptr;
  int64_t memory_size_ = -1;
#endif  // WITH_CUDA

  std::vector<Shape> input_shapes_;
  std::string FormatShapes(size_t max_num_to_format = 4);
};

}  // namespace profiler
}  // namespace oneflow

#endif  // ONEFLOW_CORE_PROFILER_EVENT_H_
