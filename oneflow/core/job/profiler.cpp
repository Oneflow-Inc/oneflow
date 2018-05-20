#include "oneflow/core/job/profiler.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/actor/act_event_logger.h"

namespace oneflow {

namespace {
struct ActTimeInfo {
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
}  // namespace

void Profiler::Profile(const Plan& plan, const std::string& act_event_filepath) {
  HashMap<int64_t, TaskType> task_id2task_type;
  for (const TaskProto& task : plan.task()) {
    CHECK(task_id2task_type.emplace(task.task_id(), task.task_type()).second);
  }

  auto act_events = of_make_unique<std::list<ActEvent>>();
  ParseActEvents(act_event_filepath, act_events.get());

  HashMap<int64_t, std::vector<ActTimeInfo>> actor_id2act_time_info;
  for (auto& act_event : *act_events.get()) {
    int64_t actor_id = act_event.actor_id();
    ActTimeInfo act_time_info(
        {act_event.ready_time(), act_event.start_time(), act_event.stop_time()});
    std::vector<ActTimeInfo>& act_time_infos = actor_id2act_time_info[actor_id];
    act_time_infos.emplace_back(act_time_info);
  }

  using ProfileInfoPair = std::pair<int64_t, ActorProfileInfo>;
  std::vector<ProfileInfoPair> profile_info_vec;
  for (auto& pair : actor_id2act_time_info) {
    std::vector<ActTimeInfo>& act_time_infos = pair.second;
    std::sort(act_time_infos.begin(), act_time_infos.end(),
              [](const ActTimeInfo& lhs, const ActTimeInfo& rhs) {
                return lhs.ready_time < rhs.ready_time;
              });

    ProfileInfoPair profile_info_pair;
    profile_info_pair.first = pair.first;
    int64_t act_num = act_time_infos.size();
    double acc_act_time = 0;
    double last_ready_time = -1;
    double acc_act_interval = 0;
    for (auto& act_time_info : act_time_infos) {
      acc_act_time += (act_time_info.stop_time - act_time_info.start_time);
      if (last_ready_time < 0) {
        last_ready_time = act_time_info.ready_time;
      } else {
        acc_act_interval += (act_time_info.ready_time - last_ready_time);
        last_ready_time = act_time_info.ready_time;
      }
    }
    profile_info_pair.second.set_act_num(act_num);
    profile_info_pair.second.set_avg_act_time(acc_act_time / act_num);
    if (act_num > 1) {
      profile_info_pair.second.set_avg_act_interval(acc_act_interval / (act_num - 1));
      profile_info_vec.emplace_back(profile_info_pair);
    }
  }

  std::sort(profile_info_vec.begin(), profile_info_vec.end(),
            [](const ProfileInfoPair& lhs, const ProfileInfoPair& rhs) {
              return lhs.second.CalcBottleNeckScore() > rhs.second.CalcBottleNeckScore();
            });
  PersistentOutStream out_stream(LocalFS(), JoinPath(LogDir(), "oneflow.profile"));
  for (const ProfileInfoPair& pair : profile_info_vec) {
    out_stream << "actor_id: " << std::to_string(pair.first)
               << " act_num: " << std::to_string(pair.second.act_num())
               << " avg_act_time: " << std::to_string(pair.second.avg_act_time())
               << " avg_act_interval: " << std::to_string(pair.second.avg_act_interval())
               << " bottleneck_score: " << std::to_string(pair.second.CalcBottleNeckScore())
               << " type: " << TaskType_Name(task_id2task_type.at(pair.first)) << "\n";
  }
}

}  // namespace oneflow
