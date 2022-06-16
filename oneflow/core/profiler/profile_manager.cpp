#include "fmt/core.h"
#include "nlohmann/json.hpp"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/profiler/kineto_shim.h"
#include "oneflow/core/profiler/profile_manager.h"
#include "oneflow/core/profiler/event.h"
#include <libkineto.h>
#include <memory>
#include <unordered_map>

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
  auto trace = StopTrace();
  const auto& kineto_events = *(trace.get()->activities());
  std::set<std::shared_ptr<IEvent>> custom_events;
  std::unordered_map<std::shared_ptr<IEvent>, int64_t> corr_ids;
  for (const auto& ev_ptr : kineto_events) {
    if (ev_ptr == nullptr) { continue; }
    const auto& activity = *ev_ptr;
    // TODO: refine below codes
    if (activity.type() == libkineto::ActivityType::CUDA_RUNTIME) {
      auto ee = CustomEvent::Create(activity.name(), CustomEventType::kCudaRuntime);
      ee->StartedAt(static_cast<time_t>(activity.timestamp()));
      ee->FinishedAt(static_cast<time_t>(activity.timestamp()) + activity.duration());
      custom_events.emplace(ee);
      corr_ids[ee] = activity.correlationId();
    }
    if (activity.type() == libkineto::ActivityType::CONCURRENT_KERNEL) {
      auto ee = CustomEvent::Create(activity.name(), CustomEventType::kCudaKernel);
      ee->StartedAt(static_cast<time_t>(activity.timestamp()));
      ee->FinishedAt(static_cast<time_t>(activity.timestamp()) + activity.duration());
      custom_events.emplace(ee);
      corr_ids[ee] = activity.correlationId();
    }
  }

  std::vector<std::shared_ptr<IEvent>> events;
  while (!events_.empty()) {
    auto e = events_.front();
    events_.pop();
#if defined(WITH_CUDA)
    auto e_kernel = std::dynamic_pointer_cast<KernelEvent>(e);
    if (e_kernel) {
      std::set<int64_t> current_corr_ids;
      if (!custom_events.empty()) {
        for (const auto& x : custom_events) {
          if (e_kernel->AddChildEventIfSo(x)) { current_corr_ids.insert(corr_ids[x]); }
        }
        for (const auto& x : custom_events) {
          if (!e_kernel->HasChildEvent(x) && current_corr_ids.count(corr_ids[x])) {
            e_kernel->AddChildEvent(x);
          }
        }
        e_kernel->WalkAmongChildren(
            [&custom_events](const std::shared_ptr<IEvent>& child) { custom_events.erase(child); });
      }
    }
#endif
    events.emplace_back(e);
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