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

void to_json(json& j, const ::oneflow::profiler::Result& result) {
  j = json{{"name", result.name},
           {"avg_duration", result.avg_duration},
           {"num_called", result.num_called},
           {"all_duration", result.all_duration},
           {"event_type", result.event_type},
           {"shapes", result.shapes}};
}

void from_json(const json& j, ::oneflow::profiler::Result& result) {
  j.at("name").get_to(result.name);
  j.at("avg_duration").get_to(result.avg_duration);
  j.at("num_called").get_to(result.num_called);
  j.at("all_duration").get_to(result.all_duration);
  j.at("event_type").get_to(result.event_type);
  j.at("shapes").get_to(result.shapes);
}

}  // namespace nlohmann

namespace oneflow {

namespace profiler {

void IEvent::Start() { started_at_ = GetTimeNow(); }
void IEvent::Finish() { finished_at_ = GetTimeNow(); }
const std::string& IEvent::GetName() const { return name_; }
time_t IEvent::GetDuration() { return finished_at_ - started_at_; }

std::shared_ptr<IEvent> IEvent::Create(EventType type, const std::string& name) {
  if (type == EventType::kCustom) { return std::make_shared<CustomEvent>(name); }
  if (type == EventType::kKernel) { return std::make_shared<KernelEvent>(name); }
  return nullptr;
}

Result KernelEvent::ConvertToResult() {
  return Result(name_, GetDuration(), 1, EventType::kKernel, __FormatShapes());
}

std::string KernelEvent::Key() { return name_ + "." + __FormatShapes(); }

std::string KernelEvent::__FormatShapes() {
  if (input_shapes_.size() == 0) { return "-"; }
  std::string result("[");
  for (size_t i = 0; i < input_shapes_.size(); ++i) {
    if (i != 0) { result += ", "; }
    const std::string current_shape = input_shapes_[i].ToString();
    if (current_shape == "()") {
      result += "scalar";
    } else {
      result += current_shape;
    }
  }
  result += "]";
  return result;
}

void KernelEvent::RecordShape(const Shape& shape) { input_shapes_.emplace_back(shape); }

Result CustomEvent::ConvertToResult() {
  return Result(name_, GetDuration(), 1, EventType::kCustom, "-");
}

std::string CustomEvent::Key() { return name_; }

std::string ProfileMgr::NewEventRecorder(EventType type, const std::string& name) {
  auto recorder = std::make_shared<EventRecorder>(type, name);
  std::string recorder_key = __GetNextEventRecorderKey(name);
  event_recorders_.emplace(recorder_key, recorder);
  return recorder_key;
}

void ProfileMgr::DeleteEventRecorder(const std::string& event_recorder_key) {
  if (event_recorders_.find(event_recorder_key) != event_recorders_.end()) {
    event_recorders_.erase(event_recorder_key);
  }
}

std::string ProfileMgr::DumpResultsJson() {
  const json j = __CountResults();
  return j.dump();
}

std::vector<Result> ProfileMgr::__CountResults() {
  std::vector<std::string> event_keys_ordered;
  std::unordered_map<std::string, Result> results;

  while (!events_.empty()) {
    auto e = events_.front();
    events_.pop();
    const auto event_key = e->Key();
    if (results.find(event_key) == results.end()) {
      const auto result = e->ConvertToResult();
      event_keys_ordered.push_back(event_key);
      results.emplace(event_key, e->ConvertToResult());
    } else {
      results[event_key].Update(e->GetDuration());
    }
  }
  std::vector<Result> final_results;
  for (const auto& event_key : event_keys_ordered) { final_results.push_back(results[event_key]); }
  return final_results;
}

std::string ProfileMgr::__GetNextEventRecorderKey(const std::string& name) {
  if (event_recorders_last_id_.find(name) == event_recorders_last_id_.end()) {
    event_recorders_last_id_[name] = 0;
  } else {
    event_recorders_last_id_[name]++;
  }
  return name + "." + std::to_string(event_recorders_last_id_[name]);
}

Maybe<void> EventRecorder::RecordShape4KernelEvent(const Shape& shape) {
  auto event = std::dynamic_pointer_cast<KernelEvent>(event_);
  CHECK_NOTNULL_OR_RETURN(event) << "Current event is not a KernelEvent.";
  event->RecordShape(shape);
  return Maybe<void>::Ok();
}

}  // namespace profiler
}  // namespace oneflow