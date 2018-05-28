#include "oneflow/core/job/profiler.h"
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

void Profiler::PushAvgActInterval(int64_t actor_id, double avg_act_interval) {
  actor_id2profile_info_[actor_id].set_avg_act_interval(avg_act_interval);
}

void Profiler::PushAvgActTime(int64_t actor_id, double avg_act_time) {
  actor_id2profile_info_[actor_id].set_avg_act_time(avg_act_time);
}

void Profiler::Profile(const Plan& plan) {
  HashMap<int64_t, TaskType> task_id2task_type;
  for (const TaskProto& task : plan.task()) {
    CHECK(task_id2task_type.emplace(task.task_id(), task.task_type()).second);
  }
  using ProfileInfoPair = std::pair<int64_t, ActorProfileInfo>;
  std::vector<ProfileInfoPair> profile_info_vec(actor_id2profile_info_.begin(),
                                                actor_id2profile_info_.end());
  std::sort(profile_info_vec.begin(), profile_info_vec.end(),
            [](const ProfileInfoPair& lhs, const ProfileInfoPair& rhs) {
              return lhs.second.CalcBottleNeckScore() > rhs.second.CalcBottleNeckScore();
            });
  PersistentOutStream out_stream(LocalFS(), JoinPath(LogDir(), "oneflow.profile"));
  double mdupdt_act_interval = 0.0;
  int32_t mdupdt_task_num = 0;
  for (const ProfileInfoPair& pair : profile_info_vec) {
    if (task_id2task_type.at(pair.first) == TaskType::kNormalMdUpdt) {
      mdupdt_task_num += 1;
      mdupdt_act_interval += pair.second.avg_act_interval();
    }
  }
  out_stream << "time_of_one_batch:" << std::to_string(mdupdt_act_interval / mdupdt_task_num)
             << "\n";
  for (const ProfileInfoPair& pair : profile_info_vec) {
    out_stream << "actor_id:" << std::to_string(pair.first)
               << " avg_act_time:" << std::to_string(pair.second.avg_act_time())
               << " avg_act_interval:" << std::to_string(pair.second.avg_act_interval())
               << " bottleneck_score:" << std::to_string(pair.second.CalcBottleNeckScore())
               << " type:" << TaskType_Name(task_id2task_type.at(pair.first)) << "\n";
  }
}

}  // namespace oneflow
