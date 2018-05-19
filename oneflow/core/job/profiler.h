#ifndef ONEFLOW_CORE_JOB_PROFILER_H_
#define ONEFLOW_CORE_JOB_PROFILER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class Profiler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Profiler);
  Profiler() = default;
  ~Profiler() = default;

  void Profile(const Plan& plan, const std::string& experiment_act_event_filepath,
               const std::string& act_event_filepath);

 private:
  struct ActProfileInfo {
    double ready_time;
    double start_time;
    double stop_time;
  };
  class ActorProfileInfo {
   public:
    ActorProfileInfo() : avg_act_interval_(-1.0), avg_act_time_(-1.0) {}

    double avg_act_interval() const { return avg_act_interval_; }
    double avg_act_time() const { return avg_act_time_; }
    int64_t act_num() const { return act_num_; }

    double CalcBottleNeckScore() const { return avg_act_time_ / avg_act_interval_; }

    void set_avg_act_interval(double val) { avg_act_interval_ = val; }
    void set_avg_act_time(double val) { avg_act_time_ = val; }
    void set_act_num(int64_t val) { act_num_ = val; }

   private:
    double avg_act_interval_;
    double avg_act_time_;
    int64_t act_num_;
  };

  HashMap<int64_t, ActorProfileInfo> actor_id2profile_info_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_PROFILER_H_
