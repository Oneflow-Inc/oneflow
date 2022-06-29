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
    const std::string& name,
#if defined(WITH_CUDA)
    cudaStream_t cuda_stream, const std::function<int64_t()>& memory_size_getter,
#endif
    const ShapeGetterFuncType& shape_getter) {
  auto pmgr = Singleton<ProfileMgr>::Get();
  if (pmgr) {
#if defined(WITH_CUDA)
    if ((pmgr->use_cpu_ && (!cuda_stream)) || (pmgr->use_cuda_ && cuda_stream)) {
      auto event = KernelEvent::Create(name, pmgr->record_shapes_ ? shape_getter : nullptr);
      if (pmgr->use_cuda_ && cuda_stream) {
        event->InitCudaEventPair(cuda_stream);
        if (pmgr->record_bandwidth_) { event->SetMemorySize(memory_size_getter()); }
      }
      return std::make_shared<EventRecorder>(event);
    }
#else  // WITH_CUDA
    if (pmgr->use_cpu_) {
      return std::make_shared<EventRecorder>(
          KernelEvent::Create(name, pmgr->record_shapes_ ? shape_getter : nullptr));
    }
#endif  // WITH_CUDA
  }

  std::shared_ptr<EventRecorder> null_recorder;
  return null_recorder;
}

}  // namespace profiler
}  // namespace oneflow
