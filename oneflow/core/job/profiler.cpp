#include "oneflow/core/job/profiler.h"

namespace oneflow {

void Profiler::PushAvgActInterval(int64_t actor_id, double avg_act_interval) {
  actor_id2profile_info_[actor_id].avg_act_interval = avg_act_interval;
}

void Profiler::PushAvgActTime(int64_t actor_id, double avg_act_time) {
  actor_id2profile_info_[actor_id].avg_act_time = avg_act_time;
}

void Profiler::PrintProfileResult() { TODO(); }

}  // namespace oneflow
