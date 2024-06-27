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

#include "fmt/core.h"
#include "fmt/format.h"
#include "oneflow/core/profiler/event.h"
#include "oneflow/core/profiler/util.h"

using json = nlohmann::json;

namespace oneflow {

namespace profiler {
nlohmann::json IEvent::ToJson() { return json{{"name", name_}, {"time", GetDuration<double>()}}; }

void IEvent::SetStartedAt(double t) { started_at_ = t; }

void IEvent::SetFinishedAt(double t) { finished_at_ = t; }

void IEvent::Start() {
  SetStartedAt(GetTimeNow(
#ifdef WITH_NPU
      true
#else
      false
#endif
      ));
}

void IEvent::Finish() {
  SetFinishedAt(GetTimeNow(
#ifdef WITH_NPU
      true
#else
      false
#endif
      ));
}

bool IEvent::IsChildOf(const IEvent* e) {
  if (!e) { return false; }
  if (this == e) { return false; }
#ifdef WITH_NPU
  const auto time_unit = EventTimeUnit::kNS;
#else
  const auto time_unit = EventTimeUnit::kUS;
#endif
  return GetStartedAt<double>(time_unit) >= e->GetStartedAt<double>(time_unit)
         && GetFinishedAt<double>(time_unit) <= e->GetFinishedAt<double>(time_unit);
}

const std::string& IEvent::GetName() const { return name_; }

nlohmann::json CustomEvent::ToJson() {
  auto j = IEvent::ToJson();
  j["type"] = EventType::kCustom;
  j["custom_type"] = type_;
  return j;
}

std::shared_ptr<CustomEvent> CustomEvent::Create(const std::string& name, CustomEventType type) {
  return std::shared_ptr<CustomEvent>(new CustomEvent(name, type));
}

nlohmann::json KernelEvent::ToJson() {
  auto j = IEvent::ToJson();
  j["type"] = EventType::kOneflowKernel;
  for (const auto& desc : description_) {
    j["description"][desc.first] = {desc.second.first, desc.second.second};
  }
#if defined(WITH_CUDA) || defined(WITH_NPU)
#ifdef WITH_CUDA
  j["memory_size"] = memory_size_;
#endif  // WITH_CUDA
  if (!children_.empty()) { j["children"] = children_; }
#endif  // WITH_CUDA || WITH_NPU
  return j;
}

std::shared_ptr<KernelEvent> KernelEvent::Create(const std::string& name,
                                                 const Description& description) {
  return std::shared_ptr<KernelEvent>(new KernelEvent(name, description));
}

}  // namespace profiler
}  // namespace oneflow
