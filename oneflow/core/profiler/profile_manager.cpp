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
#include <unordered_map>
#include "fmt/core.h"
#include "nlohmann/json.hpp"
#include "oneflow/core/profiler/kineto_shim.h"
#include "oneflow/core/profiler/profile_manager.h"
#include "oneflow/core/profiler/event.h"
#if defined(WITH_CUDA)
#include <libkineto.h>
#endif  // WITH_CUDA

using json = nlohmann::json;

namespace oneflow {
namespace profiler {

std::string ProfileManager::RegisterEventRecorder(
    const std::shared_ptr<EventRecorder>& event_recorder, const std::string& name) {
  std::string recorder_key = GetNextEventRecorderKey(name);
  event_recorders_.emplace(recorder_key, event_recorder);
  return recorder_key;
}

void ProfileManager::UnregisterEventRecorder(const std::string& event_recorder_key) {
  if (event_recorders_.find(event_recorder_key) != event_recorders_.end()) {
    event_recorders_.erase(event_recorder_key);
  }
}

std::string ProfileManager::DumpResultsJson() {
  const json j = ExportEvents();
  return j.dump();
}

std::vector<std::shared_ptr<IEvent>> ProfileManager::ExportEvents() {
#if defined(WITH_CUDA)
  auto trace = StopTrace();
  const auto& kineto_events = *(trace.get()->activities());
  std::set<std::shared_ptr<IEvent>> custom_events;
  std::unordered_map<std::shared_ptr<IEvent>, int64_t> corr_ids;

  const std::vector<std::pair<libkineto::ActivityType, CustomEventType>> type_pairs = {
      {libkineto::ActivityType::CUDA_RUNTIME, CustomEventType::kCudaRuntime},
      {libkineto::ActivityType::CONCURRENT_KERNEL, CustomEventType::kCudaKernel}};

  for (const auto& evt_ptr : kineto_events) {
    if (evt_ptr == nullptr) { continue; }
    const auto& activity = *evt_ptr;
    for (auto& pair : type_pairs) {
      if (activity.type() == pair.first) {
        auto custom_event = CustomEvent::Create(activity.name(), pair.second);
        custom_event->SetStartedAt(static_cast<time_t>(activity.timestamp()));
        custom_event->SetFinishedAt(static_cast<time_t>(activity.timestamp())
                                    + activity.duration());
        custom_events.emplace(custom_event);
        corr_ids[custom_event] = activity.correlationId();
      }
    }
  }
#endif  // WITH_CUDA
  std::vector<std::shared_ptr<IEvent>> events;
  while (!events_.empty()) {
    auto evt = events_.front();
    events_.pop();
#if defined(WITH_CUDA)
    auto evt_kernel = std::dynamic_pointer_cast<KernelEvent>(evt);
    if (evt_kernel) {
      std::set<int64_t> current_corr_ids;
      if (!custom_events.empty()) {
        for (const auto& x : custom_events) {
          if (evt_kernel->AddChildEventIfSo(x)) { current_corr_ids.insert(corr_ids[x]); }
        }
        for (const auto& x : custom_events) {
          if (!evt_kernel->HasChildEvent(x) && current_corr_ids.count(corr_ids[x])) {
            evt_kernel->AddChildEvent(x);
          }
        }
        evt_kernel->WalkAmongChildren(
            [&custom_events](const std::shared_ptr<IEvent>& child) { custom_events.erase(child); });
      }
    }
#endif  // WITH_CUDA
    events.emplace_back(evt);
  }
  return events;
}

std::string ProfileManager::GetNextEventRecorderKey(const std::string& name) {
  if (event_recorders_last_id_.find(name) == event_recorders_last_id_.end()) {
    event_recorders_last_id_[name] = 0;
  } else {
    event_recorders_last_id_[name]++;
  }
  return fmt::format("{}.{}", name, event_recorders_last_id_[name]);
}

}  // namespace profiler
}  // namespace oneflow