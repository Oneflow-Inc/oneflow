#ifndef ONEFLOW_CORE_JOB_PROFILER_H_
#define ONEFLOW_CORE_JOB_PROFILER_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class Profiler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Profiler);
  Profiler() = default;
  ~Profiler() = default;

  void PushAvgActInterval(int64_t actor_id, double avg_act_interval);
  void PushAvgActTime(int64_t actor_id, double avg_act_time);

  void PrintProfileResult();

 private:
  struct ActorProfileInfo {
    ActorProfileInfo() : avg_act_interval(-1.0), avg_act_time(-1.0) {}

    double avg_act_interval;
    double avg_act_time;
  };

  HashMap<int64_t, ActorProfileInfo> actor_id2profile_info_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_PROFILER_H_
