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
           {"num_called", result.num_called_}};
}

void from_json(const json& j, ::oneflow::profiler::Result& result) {
  j.at("op_name").get_to(result.op_name_);
  j.at("avg_duration").get_to(result.avg_duration_);
  j.at("num_called").get_to(result.num_called_);
}

}  // namespace nlohmann

namespace oneflow {

namespace profiler {

std::shared_ptr<Event> ProfileMgr::StartRecord(const std::string& op_name) {
  auto event = std::make_shared<Event>(op_name);
  events_.push(event);
  event->start_at_ = GetTimeNow();
  return event;
}
void ProfileMgr::EndRecord(const std::shared_ptr<Event>& event) { event->end_at_ = GetTimeNow(); }

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
    if (results.find(e->op_name_) == results.end()) {
      op_names_ordered.push_back(e->op_name_);
      results[e->op_name_] = Result(e->op_name_, e->end_at_ - e->start_at_, 1);
    } else {
      auto& r = results[e->op_name_];
      r.avg_duration_ =
          (r.avg_duration_ * r.num_called_ + (e->end_at_ - e->start_at_)) / (r.num_called_ + 1);
      r.num_called_++;
    }
  }
  std::vector<Result> final_results;
  for (const auto& op_name : op_names_ordered) { final_results.push_back(results[op_name]); }
  return final_results;
}

}  // namespace profiler
}  // namespace oneflow