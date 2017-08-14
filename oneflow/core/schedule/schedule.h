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

#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

class Schedule {
 public:
  explicit Schedule(Session* session) : session_(session) {}

  inline const std::unordered_map<TaskInstance*, std::pair<int32_t, int32_t>>&
  instance2ended_at() const {
    return instance2ended_at_;
  }
  inline const std::unordered_map<TaskInstance*,
                                  std::unordered_map<STask*, float>>&
  start_time_gap_to_loss() const {
    return start_time_gap_to_loss_;
  };
  inline const std::unordered_map<TaskInstance*,
                                  std::unordered_map<STask*, float>>&
  end_time_gap_to_loss() const {
    return end_time_gap_to_loss_;
  };
  inline const std::unordered_map<SDevice*, int32_t>& device2ended_at() const {
    return device2ended_at_;
  }
  inline const std::unordered_map<STask*, float>& node2interval() const {
    return node2interval_;
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
  inline std::unordered_map<TaskInstance*, std::pair<int32_t, int32_t>>&
  mut_instance2ended_at() {
    return instance2ended_at_;
  }
  inline std::unordered_map<TaskInstance*, std::unordered_map<STask*, float>>&
  mut_start_time_gap_to_loss() {
    return start_time_gap_to_loss_;
  };
  inline std::unordered_map<TaskInstance*, std::unordered_map<STask*, float>>&
  mut_end_time_gap_to_loss() {
    return end_time_gap_to_loss_;
  };
  inline std::unordered_map<SDevice*, int32_t>& mut_device2ended_at() {
    return device2ended_at_;
  }
  inline std::unordered_map<STask*, float>& mut_node2interval() {
    return node2interval_;
  }
  inline float& mut_max_interval() { return max_interval_; }
  inline std::unordered_map<SRegstDesc*, float>& mut_regst_desc2duration() {
    return regst_desc2duration_;
  }
  inline std::unordered_map<SRegstDesc*, uint32_t>& mut_regst_desc2count() {
    return regst_desc2count_;
  }

  inline Session* session() const { return session_; }

 protected:
  Session* session_;
  std::unordered_map<TaskInstance*, std::pair<int32_t, int32_t>>
      instance2ended_at_;
  std::unordered_map<TaskInstance*, std::unordered_map<STask*, float>>
      start_time_gap_to_loss_;
  std::unordered_map<TaskInstance*, std::unordered_map<STask*, float>>
      end_time_gap_to_loss_;
  std::unordered_map<SDevice*, int32_t> device2ended_at_;
  std::unordered_map<STask*, float> node2interval_;
  float max_interval_ = 0.0;
  std::unordered_map<SRegstDesc*, float> regst_desc2duration_;
  std::unordered_map<SRegstDesc*, uint32_t> regst_desc2count_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SCHEDULE_H_
