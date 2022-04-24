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

#include <functional>
#include <memory>
#include <string>
#include <queue>
#include <unordered_map>
#include <vector>
#include "oneflow/core/profiler/util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

namespace profiler {

enum class EventType { kCustom, kKernel };

struct Result {
  explicit Result(const std::string& name, time_t all_duration, int64_t num_called,
                  EventType event_type, const std::string& shapes)
      : name(name),
        all_duration(all_duration),
        num_called(num_called),
        event_type(event_type),
        shapes(shapes) {
    avg_duration = all_duration / num_called;
  }

  void Update(time_t duration) {
    all_duration += duration;
    num_called += 1;
    avg_duration = all_duration / num_called;
  }

  std::string name;
  time_t avg_duration = 0;
  time_t all_duration = 0;
  int64_t num_called = 0;
  EventType event_type;
  std::string shapes;
};

class CustomEvent;
class KernelEvent;

class IEvent {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IEvent);

  IEvent() = delete;
  explicit IEvent(const std::string& name) : name_(name) {}
  virtual std::string Key() = 0;
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

class CustomEvent final : public IEvent {
 public:
  explicit CustomEvent(const std::string& custom_name) : IEvent(custom_name) {}
  std::string Key();
  Result ConvertToResult();
};

class KernelEvent final : public IEvent {
 public:
  explicit KernelEvent(const std::string& kernel_name) : IEvent(kernel_name) {}
  std::string Key();
  Result ConvertToResult();
  void RecordShape(const Shape& shape);

 private:
  std::vector<Shape> input_shapes_;
  std::string __FormatShapes();
};

class EventRecorder;

class ProfileMgr {
 public:
  friend class EventRecorder;
  ProfileMgr() = default;

  std::string RegisterEventRecorder(const std::shared_ptr<EventRecorder>& event_recorder,
                                    const std::string& name);
  void UnregisterEventRecorder(const std::string& event_recorder_key);
  std::string DumpResultsJson();

 private:
  std::queue<std::shared_ptr<IEvent>> events_;
  std::unordered_map<std::string, std::shared_ptr<EventRecorder>> event_recorders_;
  // To prevent releasing EventRecorders of the same name.
  std::unordered_map<std::string, int64_t> event_recorders_last_id_;

  std::string __GetNextEventRecorderKey(const std::string& name);
  std::vector<Result> __CountResults();
};

class EventRecorder {
 public:
  using ShapeGetterFuncType = std::function<std::vector<Shape>(void)>;

  OF_DISALLOW_COPY_AND_MOVE(EventRecorder);

  explicit EventRecorder(const std::shared_ptr<IEvent>& event) : event_(event) {
    RegisterEventToProfileMgr(event);
    event_->Start();
  }

  Maybe<void> RegisterEventToProfileMgr(const std::shared_ptr<IEvent>& event) {
    auto pmgr = Global<profiler::ProfileMgr>::Get();
    CHECK_NOTNULL_OR_RETURN(pmgr) << "ProfileMgr has not been initialized.";
    pmgr->events_.push(event_);
    return Maybe<void>::Ok();
  }

  ~EventRecorder() {
    event_->Finish();
    event_.reset();
  }
  static std::shared_ptr<EventRecorder> CreateCustomEventRecorder(const std::string& name);
  static std::shared_ptr<EventRecorder> CreateKernelEventRecorder(
      const std::string& name, const ShapeGetterFuncType& shape_getter = {});

 private:
  std::shared_ptr<IEvent> event_;
};

}  // namespace profiler
}  // namespace oneflow
#endif  // ONEFLOW_CORE_PROFILER_COLLECTION_H_
