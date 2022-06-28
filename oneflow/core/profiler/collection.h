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
#include "nlohmann/json.hpp"
#include "oneflow/core/profiler/event.h"
#include "oneflow/core/profiler/util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace profiler {

class EventRecorder;

class ProfileMgr {
 public:
  friend class EventRecorder;

  ProfileMgr(bool use_cpu, bool use_cuda, bool record_shapes, bool record_bandwidth)
      : use_cpu_(use_cpu),
        use_cuda_(use_cuda),
        record_shapes_(record_shapes),
        record_bandwidth_(record_bandwidth) {}

  std::string RegisterEventRecorder(const std::shared_ptr<EventRecorder>& event_recorder,
                                    const std::string& name);
  void UnregisterEventRecorder(const std::string& event_recorder_key);
  std::string DumpResultsJson();

 private:
  bool use_cpu_;
  bool use_cuda_;
  bool record_shapes_;
  bool record_bandwidth_;

  std::queue<std::shared_ptr<IEvent>> events_;
  std::unordered_map<std::string, std::shared_ptr<EventRecorder>> event_recorders_;
  // To prevent releasing EventRecorders of the same name.
  std::unordered_map<std::string, int64_t> event_recorders_last_id_;

  std::string GetNextEventRecorderKey(const std::string& name);
  std::vector<std::shared_ptr<IEvent>> ExportEvents();
};

class EventRecorder {
 public:
  using ShapeGetterFuncType = std::function<std::vector<Shape>(void)>;

  OF_DISALLOW_COPY_AND_MOVE(EventRecorder);

  explicit EventRecorder(const std::shared_ptr<IEvent>& event) : event_(event) {
    CHECK_JUST(RegisterEventToProfileMgr(event));
    event_->Start();
  }

  Maybe<void> RegisterEventToProfileMgr(const std::shared_ptr<IEvent>& event) {
    auto* pmgr = JUST(SingletonMaybe<ProfileMgr>());
    pmgr->events_.push(event_);
    return Maybe<void>::Ok();
  }

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
      cudaStream_t cuda_stream, const std::function<int64_t()>& memory_size_getter,
#endif
      const ShapeGetterFuncType& shape_getter);

 private:
  std::shared_ptr<IEvent> event_;
};

}  // namespace profiler
}  // namespace oneflow
#endif  // ONEFLOW_CORE_PROFILER_COLLECTION_H_
