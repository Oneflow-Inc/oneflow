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

#include <functional>
#include <memory>
#include <vector>
#include "nlohmann/json.hpp"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {

namespace profiler {

class ProfileManager;

enum class EventType {
  kCustom,        // has three kinds
  kOneflowKernel  // OneFlow cpu/cuda kernel
};
enum class CustomEventType {
  kDefault,     // for record_function
  kCudaKernel,  // cuda kernel
  kCudaRuntime  // something like cudaLaunchKernel
};
enum class EventTimeUnit { kNS, kUS };

class IEvent {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IEvent);

  IEvent() = delete;
  IEvent(const std::string& name, EventTimeUnit time_unit) : name_(name), time_unit_(time_unit) {}

  virtual nlohmann::json ToJson();
  virtual ~IEvent() = default;

  virtual void Start();
  virtual void Finish();
  bool IsChildOf(const IEvent* e);

  const std::string& GetName() const;
  template<typename T>
  const T GetDuration(EventTimeUnit time_unit = EventTimeUnit::kUS) const;
  template<typename T>
  const T GetStartedAt(EventTimeUnit time_unit = EventTimeUnit::kUS) const;
  template<typename T>
  const T GetFinishedAt(EventTimeUnit time_unit = EventTimeUnit::kUS) const;

 protected:
  virtual void SetStartedAt(double t);
  virtual void SetFinishedAt(double t);

  std::string name_;
  EventTimeUnit time_unit_;
  double started_at_ = 0;
  double finished_at_ = 0;
};

inline double ConvertTime(double time_, EventTimeUnit src_time_unit, EventTimeUnit dst_time_unit) {
  if (src_time_unit == EventTimeUnit::kNS && dst_time_unit == EventTimeUnit::kUS) {
    return time_ / 1000;
  }
  if (src_time_unit == EventTimeUnit::kUS && dst_time_unit == EventTimeUnit::kNS) {
    return time_ * 1000;
  }
  return time_;
}

template<>
const inline double IEvent::GetStartedAt<double>(EventTimeUnit time_unit) const {
  return ConvertTime(started_at_, time_unit_, time_unit);
}

template<>
const inline time_t IEvent::GetStartedAt<time_t>(EventTimeUnit time_unit) const {
  return static_cast<time_t>(GetStartedAt<double>(time_unit));
}

template<>
const inline double IEvent::GetFinishedAt<double>(EventTimeUnit time_unit) const {
  return ConvertTime(finished_at_, time_unit_, time_unit);
}

template<>
const inline time_t IEvent::GetFinishedAt<time_t>(EventTimeUnit time_unit) const {
  return static_cast<time_t>(GetFinishedAt<double>(time_unit));
}

template<>
const inline double IEvent::GetDuration<double>(EventTimeUnit time_unit) const {
  return GetFinishedAt<double>(time_unit) - GetStartedAt<double>(time_unit);
}

template<>
const inline time_t IEvent::GetDuration<time_t>(EventTimeUnit time_unit) const {
  return static_cast<time_t>(GetDuration<double>(time_unit));
}

class CustomEvent final : public IEvent {
 public:
  friend class ProfileManager;

  nlohmann::json ToJson() override;

  static std::shared_ptr<CustomEvent> Create(const std::string& name,
                                             CustomEventType type = CustomEventType::kDefault);

 private:
  CustomEventType type_;
  CustomEvent(const std::string& custom_name, CustomEventType type)
      : IEvent(custom_name,
               type == CustomEventType::kDefault ? EventTimeUnit::kNS : EventTimeUnit::kUS),
        type_(type) {}
};

class KernelEvent final : public IEvent {
 public:
  using Description = std::map<std::string, std::pair<std::string, int64_t>>;

  nlohmann::json ToJson() override;

  static std::shared_ptr<KernelEvent> Create(const std::string& name,
                                             const Description& description);

#if defined(WITH_CUDA)
  void SetMemorySize(int64_t memory_size) { memory_size_ = memory_size; }
  void AddChildEvent(const std::shared_ptr<IEvent>& e) { children_.emplace(e); }
  bool AddChildEventIfSo(const std::shared_ptr<IEvent>& e) {
    if (e->IsChildOf(dynamic_cast<IEvent*>(this))) {
      children_.emplace(e);
      return true;
    }
    return false;
  }
  bool HasChildEvent(const std::shared_ptr<IEvent>& e) { return children_.count(e); }
  void WalkAmongChildren(const std::function<void(const std::shared_ptr<IEvent>& e)>& f) const {
    for (const auto& x : children_) { f(x); }
  }
#endif  // WITH_CUDA

 private:
  KernelEvent(const std::string& kernel_name, const Description& description)
      : IEvent(kernel_name, EventTimeUnit::kNS), description_(description) {}

#if defined(WITH_CUDA)
  int64_t memory_size_ = -1;
  std::set<std::shared_ptr<IEvent>> children_;
#endif  // WITH_CUDA

  const Description description_;
};

}  // namespace profiler
}  // namespace oneflow

namespace nlohmann {

inline void to_json(json& j, const std::shared_ptr<::oneflow::profiler::IEvent>& event) {
  j = event->ToJson();
}

}  // namespace nlohmann

#endif  // ONEFLOW_CORE_PROFILER_EVENT_H_
