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

#include <memory>
#include <string>
#include <queue>
#include <unordered_map>
#include "oneflow/core/profiler/util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/global.h"

namespace oneflow {

namespace profiler {

struct Result {
  Result() = default;

  explicit Result(const std::string& op_name, time_t all_duration, int64_t num_called)
      : op_name_(op_name), all_duration_(all_duration), num_called_(num_called) {
    avg_duration_ = all_duration / num_called;
  }

  void Update(time_t duration) {
    all_duration_ += duration;
    num_called_ += 1;
    avg_duration_ = all_duration_ / num_called_;
  }

  std::string op_name_;
  time_t avg_duration_ = 0;
  time_t all_duration_ = 0;
  int64_t num_called_ = 0;
};

enum class EventType { kCustom, kKernel };
class CustomEvent;
class KernelEvent;

class IEvent {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IEvent);

  IEvent() = delete;
  explicit IEvent(const std::string& name) : name_(name) {}
  virtual Result ConvertToResult() = 0;
  virtual ~IEvent() = default;

  void Start();
  void Finish();
  const std::string& GetName() const;
  time_t GetDuration();
  static std::shared_ptr<IEvent> Create(EventType type, const std::string& name);

 protected:
  std::string name_;
  time_t started_at_ = 0;
  time_t finished_at_ = 0;
};

class CustomEvent : public IEvent {
 public:
  explicit CustomEvent(const std::string& custom_name) : IEvent(custom_name) {}
  Result ConvertToResult();
};

class KernelEvent : public IEvent {
 public:
  explicit KernelEvent(const std::string& kernel_name) : IEvent(kernel_name) {}
  Result ConvertToResult();
};

class EventRecorder;

class ProfileMgr {
 public:
  friend class EventRecorder;
  ProfileMgr() = default;
  std::shared_ptr<EventRecorder> NewEventRecorder(EventType type, const std::string& name);
  void DeleteEventRecorder(const std::string& name);
  std::string DumpResultsJson();

 private:
  std::queue<std::shared_ptr<IEvent>> events_;
  std::unordered_map<std::string, std::shared_ptr<EventRecorder>> event_recorders_;

  std::vector<Result> __CountResults();
};

class EventRecorder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EventRecorder);

  explicit EventRecorder(EventType type, const std::string& name) {
    event_ = IEvent::Create(type, name);
    auto pmgr = Global<profiler::ProfileMgr>::Get();
    if (pmgr) { pmgr->events_.push(event_); }
    event_->Start();
  }
  ~EventRecorder() {
    event_->Finish();
    event_.reset();
  }

 private:
  std::shared_ptr<IEvent> event_;
};

}  // namespace profiler
}  // namespace oneflow
#endif  // ONEFLOW_CORE_PROFILER_COLLECTION_H_
