/**
 * Copyright 2017 Xinqi Li
 */
#ifndef ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_SCHEDULE_RESULT_H_
#define ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_SCHEDULE_RESULT_H_

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

#include "oneflow/core/schedule/node.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

class ScheduleResult {
 public:
  ScheduleResult() {}
  DEFINE_PURE_VIRTUAL_TYPE();

  inline const std::unordered_map<Arc*, std::pair<int32_t, int32_t>>&
  instance2ended_at() const {
    return instance2ended_at_;
  }
  inline std::unordered_map<Arc*, std::pair<int32_t, int32_t>>&
  mut_instance2ended_at() {
    return instance2ended_at_;
  }
  inline const std::unordered_map<Arc*, std::unordered_map<Node*, float>>&
  start_time_gap_to_loss() const {
    return start_time_gap_to_loss_;
  };
  inline std::unordered_map<Arc*, std::unordered_map<Node*, float>>&
  mut_start_time_gap_to_loss() {
    return start_time_gap_to_loss_;
  };
  inline const std::unordered_map<Arc*, std::unordered_map<Node*, float>>&
  end_time_gap_to_loss() const {
    return end_time_gap_to_loss_;
  };
  inline std::unordered_map<Arc*, std::unordered_map<Node*, float>>&
  mut_end_time_gap_to_loss() {
    return end_time_gap_to_loss_;
  };
  inline const std::unordered_map<Node*, int32_t>& device2ended_at() const {
    return device2ended_at_;
  }
  inline std::unordered_map<Node*, int32_t>& mut_device2ended_at() {
    return device2ended_at_;
  }

  inline const std::unordered_map<Node*, float>& node2interval() const {
    return node2interval_;
  }
  inline std::unordered_map<Node*, float>& mut_node2interval() {
    return node2interval_;
  }
  inline float& mut_max_interval() { return max_interval_; }
  inline const float max_interval() const { return max_interval_; }

  inline const std::unordered_map<Node*, float>& regst_desc2duration() const {
    return regst_desc2duration_;
  }
  inline std::unordered_map<Node*, float>& mut_regst_desc2duration() {
    return regst_desc2duration_;
  }
  inline const std::unordered_map<Node*, int32_t>& regst_desc2count() const {
    return regst_desc2count_;
  }
  inline std::unordered_map<Node*, int32_t>& mut_regst_desc2count() {
    return regst_desc2count_;
  }

 protected:
  std::unordered_map<Arc*, std::pair<int32_t, int32_t>> instance2ended_at_;
  std::unordered_map<Arc*, std::unordered_map<Node*, float>>
      start_time_gap_to_loss_;
  std::unordered_map<Arc*, std::unordered_map<Node*, float>>
      end_time_gap_to_loss_;
  std::unordered_map<Node*, int32_t> device2ended_at_;
  std::unordered_map<Node*, float> node2interval_;
  float max_interval_ = 0.0;
  std::unordered_map<Node*, float> regst_desc2duration_;
  std::unordered_map<Node*, int32_t> regst_desc2count_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_SCHEDULE_RESULT_H_
