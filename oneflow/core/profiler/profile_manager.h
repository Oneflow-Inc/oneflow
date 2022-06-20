#ifndef ONEFLOW_CORE_PROFILER_PROFILE_MANAGER_H_
#define ONEFLOW_CORE_PROFILER_PROFILE_MANAGER_H_

#include <memory>
#include <queue>
#include <set>
#include <unordered_map>
#include "oneflow/core/profiler/kineto_shim.h"

namespace oneflow {
namespace profiler {

class IEvent;
class EventRecorder;

class ProfileManager {
 public:
  friend class EventRecorder;

  ProfileManager(bool use_cpu, bool use_cuda, bool record_shapes, bool record_bandwidth)
      : use_cpu_(use_cpu),
        use_cuda_(use_cuda),
        record_shapes_(record_shapes),
        record_bandwidth_(record_bandwidth) {
    std::set<ActivityType> activities{};
    if (use_cpu) { activities.insert(ActivityType::CPU); }
    if (use_cuda) { activities.insert(ActivityType::CUDA); }
    PrepareTrace(false, activities);
    StartTrace();
  }

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

}  // namespace profiler
}  // namespace oneflow

#endif // ONEFLOW_CORE_PROFILER_PROFILE_MANAGER_H_
