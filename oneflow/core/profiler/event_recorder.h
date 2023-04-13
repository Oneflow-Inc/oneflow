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
#ifndef ONEFLOW_CORE_PROFILER_EVENT_RECORDER_H_
#define ONEFLOW_CORE_PROFILER_EVENT_RECORDER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/profiler/event.h"

namespace oneflow {
namespace profiler {

class EventRecorder {
 public:
  using DescriptionGetter = std::function<std::pair<std::string, int64_t>()>;

  OF_DISALLOW_COPY_AND_MOVE(EventRecorder);

  explicit EventRecorder(const std::shared_ptr<IEvent>& event) : event_(event) {
    CHECK_JUST(RegisterEventToProfileManager(event));
    event_->Start();
  }

  Maybe<void> RegisterEventToProfileManager(const std::shared_ptr<IEvent>& event);

  ~EventRecorder() {
    if (event_) {
      event_->Finish();
      event_.reset();
    }
  }

  static std::shared_ptr<EventRecorder> CreateCustomEventRecorder(const std::string& name);

  static Maybe<EventRecorder> CreateKernelEventRecorder(
      const std::string& name,
#if defined(WITH_CUDA)
      const std::function<int64_t()>& memory_size_getter,
#endif
      const DescriptionGetter& input_shapes_getter, const DescriptionGetter& attrs_getter);

 private:
  std::shared_ptr<IEvent> event_;
};

}  // namespace profiler
}  // namespace oneflow

#endif  // ONEFLOW_CORE_PROFILER_EVENT_RECORDER_H_
