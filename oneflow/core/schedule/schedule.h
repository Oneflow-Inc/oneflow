#ifndef ONEFLOW_CORE_SCHEDULE_SCHEDULE_H_
#define ONEFLOW_CORE_SCHEDULE_SCHEDULE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"

namespace oneflow {
namespace schedule {

class Schedule {
 public:
  explicit Schedule(const Session& session) : session_(&session) {}
  virtual ~Schedule() = default;

  void Clear();
  void UpdateDuration();
  void UpdateRegstCount();
  void UpdateInterval();
  float GetDuration(const TaskInstance* src_node, const TaskInstance* dst_node);

  //	getter
  inline const Session& session() const { return *session_; }
  inline const SGraph& sgraph() const { return session_->sgraph(); }
  inline const std::unordered_map<const TaskInstance*, std::pair<float, float>>&
  instance2ended_at() const {
    return instance2ended_at_;
  }
  inline const std::unordered_map<const SDevice*, float>& device2ended_at()
      const {
    return device2ended_at_;
  }
  inline const float max_interval() const { return max_interval_; }

  inline const std::unordered_map<const SRegstDesc*, float>&
  regst_desc2duration() const {
    return regst_desc2duration_;
  }
  inline const float GetRegstDescDuration(const SRegstDesc* regst_desc) const {
    return GetOrDefault(regst_desc2duration(), regst_desc,
                        static_cast<float>(0));
  }
  inline const std::unordered_map<const SRegstDesc*, uint32_t>&
  regst_desc2count() const {
    return regst_desc2count_;
  }
  inline const uint32_t max_regst_count() {
    uint32_t max_count = 0;
    for (const auto& p : regst_desc2count_) {
      max_count = std::max(max_count, p.second);
    }
    return max_count;
  }

  //	setter
  inline std::unordered_map<const TaskInstance*, std::pair<float, float>>&
  mut_instance2ended_at() {
    return instance2ended_at_;
  }
  inline std::unordered_map<const SDevice*, float>& mut_device2ended_at() {
    return device2ended_at_;
  }
  inline float& mut_max_interval() { return max_interval_; }
  inline std::unordered_map<const SRegstDesc*, float>&
  mut_regst_desc2duration() {
    return regst_desc2duration_;
  }
  inline std::unordered_map<const SRegstDesc*, uint32_t>&
  mut_regst_desc2count() {
    return regst_desc2count_;
  }

  void PrintRegstNum();
  void PrintSchedule();

 protected:
  float GetInitiationIntervalFromIntervals(const std::vector<float>& intervals);
  const Session* session_;
  std::unordered_map<const TaskInstance*, std::pair<float, float>>
      instance2ended_at_;
  std::unordered_map<const SDevice*, float> device2ended_at_;
  float max_interval_ = 0.0;
  std::unordered_map<const SRegstDesc*, float> regst_desc2duration_;
  std::unordered_map<const SRegstDesc*, uint32_t> regst_desc2count_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SCHEDULE_H_
