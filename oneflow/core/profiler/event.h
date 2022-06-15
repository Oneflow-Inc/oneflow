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

#include <bits/types/time_t.h>
#include <functional>
#include <memory>
#include <vector>
#include "nlohmann/json.hpp"
#include "oneflow/core/common/util.h"

namespace oneflow {

class Shape;

namespace profiler {

enum class EventType { kCustom, kOneflowKernel };
enum class CustomEventType { kDefault, kCudaKernel, kCudaRuntime };

class IEvent {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IEvent);

  IEvent() = delete;
  explicit IEvent(const std::string& name) : name_(name) {}

  virtual std::string Key() = 0;
  virtual nlohmann::json ToJson();
  virtual ~IEvent() = default;
  // Make sure you know what you are doing when using StartedAt and FinishedAt
  virtual void StartedAt(time_t t);
  virtual void FinishedAt(time_t t);
  virtual void Start();
  virtual void Finish();
  bool IsChildOf(const std::shared_ptr<IEvent>& e);

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

  static std::shared_ptr<CustomEvent> Create(const std::string& name,
                                             CustomEventType type = CustomEventType::kDefault);

 private:
  CustomEventType type_;
  explicit CustomEvent(const std::string& custom_name, CustomEventType type)
      : IEvent(custom_name), type_(type) {}
};

class KernelEvent final : public IEvent {
 public:
  std::string Key() override;

  nlohmann::json ToJson() override;

  static std::shared_ptr<KernelEvent> Create(
      const std::string& name, const std::function<std::vector<Shape>(void)>& shape_getter);

  void RecordShape(const Shape& shape);

#if defined(WITH_CUDA)
  void SetMemorySize(int64_t memory_size) { memory_size_ = memory_size; }
  void AddChild(const std::shared_ptr<IEvent>& e) { children_.emplace_back(e); }
  void WalkAmongChildren(const std::function<void(const std::shared_ptr<IEvent>& e)>& f) const {
    for (const auto& x : children_) { f(x); }
  }
#endif  // WITH_CUDA

 private:
  explicit KernelEvent(const std::string& kernel_name,
                       const std::function<std::vector<Shape>(void)>& shape_getter)
      : IEvent(kernel_name) {
    if (shape_getter) { input_shapes_ = shape_getter(); }
  }

#if defined(WITH_CUDA)
  int64_t memory_size_ = -1;
  std::vector<std::shared_ptr<IEvent>> children_;
#endif  // WITH_CUDA

  std::vector<Shape> input_shapes_;
  std::string GetFormatedInputShapes(size_t max_num_to_format = 4);
};

}  // namespace profiler
}  // namespace oneflow

namespace nlohmann {

inline void to_json(json& j, const std::shared_ptr<::oneflow::profiler::IEvent>& event) {
  j = event->ToJson();
}

}  // namespace nlohmann

#endif  // ONEFLOW_CORE_PROFILER_EVENT_H_
