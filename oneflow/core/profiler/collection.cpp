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
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include "nlohmann/json.hpp"
#include "oneflow/core/profiler/collection.h"
#include "oneflow/core/profiler/util.h"

using json = nlohmann::json;

namespace nlohmann {

void to_json(json& j, const std::shared_ptr<::oneflow::profiler::IEvent>& event) {
  j = event->ToJson();
}

}  // namespace nlohmann

namespace oneflow {

namespace profiler {

nlohmann::json IEvent::ToJson() {
  return json{{"name", name_},
              {"cpu_time", static_cast<double>(GetDuration()) / 1000},
              {"input_shapes", "-"}};
}

void IEvent::Start() { started_at_ = GetTimeNow(); }

void IEvent::Finish() { finished_at_ = GetTimeNow(); }

const std::string& IEvent::GetName() const { return name_; }

time_t IEvent::GetDuration() { return finished_at_ - started_at_; }

nlohmann::json KernelEvent::ToJson() {
  auto j = IEvent::ToJson();
  j["type"] = EventType::kKernel;
  j["input_shapes"] = FormatShapes();
  return j;
}

std::string KernelEvent::Key() { return name_ + "." + FormatShapes(); }

std::string KernelEvent::FormatShapes(size_t max_num_to_format) {
  if (input_shapes_.size() == 0) { return "-"; }
  std::string result("[");
  for (size_t i = 0; i < std::min(input_shapes_.size(), max_num_to_format); ++i) {
    if (i != 0) { result += ", "; }
    const std::string current_shape = input_shapes_[i].ToString();
    if (current_shape == "()") {
      result += "scalar";
    } else {
      result += current_shape;
    }
  }
  if (input_shapes_.size() > max_num_to_format) { result += ", ..."; }
  result += "]";
  return result;
}

void KernelEvent::RecordShape(const Shape& shape) { input_shapes_.emplace_back(shape); }

void KernelEvent::Start() {
  if (device_ == KernelEventDevice::kCPU) {
    IEvent::Start();
  } else {
  }
}

void KernelEvent::Finish() {
  if (device_ == KernelEventDevice::kCPU) {
    IEvent::Finish();
  } else {
  }
}

std::shared_ptr<IEvent> KernelEvent::Create(
    const std::string& name, KernelEventDevice device,
    const std::function<std::vector<Shape>(void)>& shape_getter) {
  return std::make_shared<KernelEvent>(name, device, shape_getter);
}

nlohmann::json CustomEvent::ToJson() {
  auto j = IEvent::ToJson();
  j["type"] = EventType::kKernel;
  return j;
}

std::string CustomEvent::Key() { return name_; }

std::shared_ptr<IEvent> CustomEvent::Create(const std::string& name) {
  return std::make_shared<CustomEvent>(name);
}

std::string ProfileMgr::RegisterEventRecorder(const std::shared_ptr<EventRecorder>& event_recorder,
                                              const std::string& name) {
  std::string recorder_key = GetNextEventRecorderKey(name);
  event_recorders_.emplace(recorder_key, event_recorder);
  return recorder_key;
}
void ProfileMgr::UnregisterEventRecorder(const std::string& event_recorder_key) {
  if (event_recorders_.find(event_recorder_key) != event_recorders_.end()) {
    event_recorders_.erase(event_recorder_key);
  }
}

std::string ProfileMgr::DumpResultsJson() {
  const json j = ExportEvents();
  return j.dump();
}

std::vector<std::shared_ptr<IEvent>> ProfileMgr::ExportEvents() {
  std::vector<std::shared_ptr<IEvent>> events;
  while (!events_.empty()) {
    auto e = events_.front();
    events_.pop();
    events.emplace_back(e);
  }
  return events;
}

std::string ProfileMgr::GetNextEventRecorderKey(const std::string& name) {
  if (event_recorders_last_id_.find(name) == event_recorders_last_id_.end()) {
    event_recorders_last_id_[name] = 0;
  } else {
    event_recorders_last_id_[name]++;
  }
  return name + "." + std::to_string(event_recorders_last_id_[name]);
}

std::shared_ptr<EventRecorder> EventRecorder::CreateCustomEventRecorder(const std::string& name) {
  return std::make_shared<EventRecorder>(CustomEvent::Create(name));
}

Maybe<EventRecorder> EventRecorder::CreateKernelEventRecorder(
    const std::string& name, KernelEventDevice device, const ShapeGetterFuncType& shape_getter) {
  auto pmgr = Global<ProfileMgr>::Get();
  if (pmgr) {
    if (pmgr->use_cpu_ && device == KernelEventDevice::kCPU) {
      return std::make_shared<EventRecorder>(KernelEvent::Create(
          name, KernelEventDevice::kCPU, pmgr->record_shapes_ ? shape_getter : nullptr));
    }
    if (pmgr->use_cuda_ && device == KernelEventDevice::kCUDA) {
      return std::make_shared<EventRecorder>(KernelEvent::Create(
          name, KernelEventDevice::kCUDA, pmgr->record_shapes_ ? shape_getter : nullptr));
    }
  }

  std::shared_ptr<EventRecorder> null_recorder;
  return null_recorder;
}

}  // namespace profiler
}  // namespace oneflow