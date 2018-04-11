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

  void Profile();

 private:
  class ActorProfileInfo {
   public:
    ActorProfileInfo() : avg_act_interval_(-1.0), avg_act_time_(-1.0) {}

    double avg_act_interval() const { return avg_act_interval_; }
    double avg_act_time() const { return avg_act_time_; }

    double CalcBottleNeckScore() const {
      return avg_act_time_ / avg_act_interval_;
    }

    void set_avg_act_interval(double val) { avg_act_interval_ = val; }
    void set_avg_act_time(double val) { avg_act_time_ = val; }

   private:
    double avg_act_interval_;
    double avg_act_time_;
  };

  HashMap<int64_t, ActorProfileInfo> actor_id2profile_info_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_PROFILER_H_
