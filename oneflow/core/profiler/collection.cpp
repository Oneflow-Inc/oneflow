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
#include "nlohmann/json.hpp"
#include "oneflow/core/profiler/collection.h"
#include "oneflow/core/profiler/util.h"

using json = nlohmann::json;

namespace nlohmann {

void to_json(json& j, const ::oneflow::profiler::Result& result) {
  j = json{{"op_name", result.op_name_},
           {"avg_duration", result.avg_duration_},
           {"num_called", result.num_called_},
           {"all_duration", result.all_duration_}};
}

void from_json(const json& j, ::oneflow::profiler::Result& result) {
  j.at("op_name").get_to(result.op_name_);
  j.at("avg_duration").get_to(result.avg_duration_);
  j.at("num_called").get_to(result.num_called_);
  j.at("all_duration").get_to(result.all_duration_);
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

Result KernelEvent::ConvertToResult() { return Result(name_, GetDuration(), 1); }

Result CustomEvent::ConvertToResult() { return Result(name_, GetDuration(), 1); }

std::shared_ptr<EventRecorder> ProfileMgr::NewEventRecorder(EventType type,
                                                            const std::string& name) {
  auto recorder = std::make_shared<EventRecorder>(type, name);
  event_recorders_.emplace(name, recorder);
  return recorder;
}

void ProfileMgr::DeleteEventRecorder(const std::string& name) {
  if (event_recorders_.find(name) != event_recorders_.end()) { event_recorders_.erase(name); }
}

std::string ProfileMgr::DumpResultsJson() {
  const json j = __CountResults();
  return j.dump();
}

std::vector<Result> ProfileMgr::__CountResults() {
  std::vector<std::string> op_names_ordered;
  std::map<std::string, Result> results;
  while (!events_.empty()) {
    auto e = events_.front();
    events_.pop();
    const auto& event_name = e->GetName();
    if (results.find(event_name) == results.end()) {
      op_names_ordered.push_back(event_name);
      results[event_name] = e->ConvertToResult();
    } else {
      results[event_name].Update(e->GetDuration());
    }
  }
  std::vector<Result> final_results;
  for (const auto& op_name : op_names_ordered) { final_results.push_back(results[op_name]); }
  return final_results;
}

}  // namespace profiler
}  // namespace oneflow