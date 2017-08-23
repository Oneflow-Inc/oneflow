/**
 * Copyright 2017 Xinqi Li
 */
#ifndef ONEFLOW_CORE_SCHEDULE_SCHEDULE_H_
#define ONEFLOW_CORE_SCHEDULE_SCHEDULE_H_

#include <limits.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"

namespace oneflow {
namespace schedule {

class Schedule {
 public:
  explicit Schedule(const Session* session) : session_(session) {}

  void Clear();
  void UpdateDuration();
  void UpdateRegstCount();
  void UpdateInterval();
  float GetDuration(TaskInstance* src_node, TaskInstance* dst_node);

  //	getter
  inline const Session* session() const { return session_; }
  inline const std::unordered_map<TaskInstance*, std::pair<float, float>>&
  instance2ended_at() const {
    return instance2ended_at_;
  }
  inline const std::unordered_map<SDevice*, float>& device2ended_at() const {
    return device2ended_at_;
  }
  inline const float max_interval() const { return max_interval_; }

  inline const std::unordered_map<SRegstDesc*, float>& regst_desc2duration()
      const {
    return regst_desc2duration_;
  }
  inline const std::unordered_map<SRegstDesc*, uint32_t>& regst_desc2count()
      const {
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
  inline std::unordered_map<TaskInstance*, std::pair<float, float>>&
  mut_instance2ended_at() {
    return instance2ended_at_;
  }
  inline std::unordered_map<SDevice*, float>& mut_device2ended_at() {
    return device2ended_at_;
  }
  inline float& mut_max_interval() { return max_interval_; }
  inline std::unordered_map<SRegstDesc*, float>& mut_regst_desc2duration() {
    return regst_desc2duration_;
  }
  inline std::unordered_map<SRegstDesc*, uint32_t>& mut_regst_desc2count() {
    return regst_desc2count_;
  }

  void PrintRegstNum();
  void PrintSchedule();

 protected:
  const Session* session_;
  std::unordered_map<TaskInstance*, std::pair<float, float>> instance2ended_at_;
  std::unordered_map<SDevice*, float> device2ended_at_;
  float max_interval_ = 0.0;
  std::unordered_map<SRegstDesc*, float> regst_desc2duration_;
  std::unordered_map<SRegstDesc*, uint32_t> regst_desc2count_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SCHEDULE_H_
