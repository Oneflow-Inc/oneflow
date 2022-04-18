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

#ifndef ONEFLOW_CORE_PROFILER_COLLECTION_H_
#define ONEFLOW_CORE_PROFILER_COLLECTION_H_

#include <bits/stdint-intn.h>
#include <memory>
#include <string>
#include <queue>
#include "oneflow/core/profiler/util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/global.h"

namespace oneflow {

namespace profiler {

struct Event {
  Event() = default;
  explicit Event(const std::string& op_name) : op_name_(op_name) {}

  std::string op_name_;
  time_t start_at_ = 0;
  time_t end_at_ = 0;
};

struct Result {
  Result() = default;
  explicit Result(const std::string& op_name, time_t avg_duration, int64_t num_called)
      : op_name_(op_name), avg_duration_(avg_duration), num_called_(num_called) {}

  std::string op_name_;
  time_t avg_duration_ = 0;
  int64_t num_called_ = 0;
};

class ProfileMgr {
 public:
  ProfileMgr() = default;
  std::shared_ptr<Event> StartRecord(const std::string& op_name);
  void EndRecord(const std::shared_ptr<Event>& event);
  std::string DumpResultsJson();

 private:
  std::queue<std::shared_ptr<Event>> events_;
  std::vector<Result> __CountResults();
};

class EventRecorder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EventRecorder);

  explicit EventRecorder(const std::string& name) {
    auto pmgr = Global<profiler::ProfileMgr>::Get();
    if (pmgr) { event_ = pmgr->StartRecord(name); }
  }
  ~EventRecorder() {
    auto pmgr = Global<profiler::ProfileMgr>::Get();
    if (pmgr) { pmgr->EndRecord(event_); }
  }

 private:
  std::shared_ptr<Event> event_;
};

}  // namespace profiler
}  // namespace oneflow
#endif  // ONEFLOW_CORE_PROFILER_COLLECTION_H_
