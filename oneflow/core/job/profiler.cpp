#include "oneflow/core/job/profiler.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/actor/act_event_logger.h"

namespace oneflow {

void Profiler::Profile(const Plan& plan, const std::string& experiment_act_event_filepath,
                       const std::string& act_event_filepath) {
  HashMap<int64_t, TaskType> task_id2task_type;
  for (const TaskProto& task : plan.task()) {
    CHECK(task_id2task_type.emplace(task.task_id(), task.task_type()).second);
  }

  // Get experiment avg_act_time
  auto experiment_act_events = of_make_unique<std::list<ActEvent>>();
  ParseActEvents(experiment_act_event_filepath, experiment_act_events.get());
  HashMap<int64_t, std::vector<ActProfileInfo>> experiment_actor_id2act_profile_info;
  for (auto& act_event : *experiment_act_events.get()) {
    int64_t actor_id = act_event.actor_id();
    ActProfileInfo act_profile_info(
        {act_event.ready_time(), act_event.start_time(), act_event.stop_time()});
    std::vector<ActProfileInfo>& act_profile_infos = experiment_actor_id2act_profile_info[actor_id];
    act_profile_infos.emplace_back(act_profile_info);
  }
  HashMap<int64_t, ActorProfileInfo> experiment_actor_id2profile_info;
  for (auto& pair : experiment_actor_id2act_profile_info) {
    std::vector<ActProfileInfo>& act_profile_infos = pair.second;
    std::sort(act_profile_infos.begin(), act_profile_infos.end(),
              [](const ActProfileInfo& lhs, const ActProfileInfo& rhs) {
                return lhs.ready_time < rhs.ready_time;
              });
    ActorProfileInfo actor_profile_info;
    int64_t act_num = act_profile_infos.size();
    double acc_act_time = 0;
    double last_ready_time = -1;
    double acc_act_interval = 0;
    for (auto& act_profile_info : act_profile_infos) {
      acc_act_time += (act_profile_info.stop_time - act_profile_info.start_time);
      if (last_ready_time < 0) {
        last_ready_time = act_profile_info.ready_time;
      } else {
        acc_act_interval += (act_profile_info.ready_time - last_ready_time);
        last_ready_time = act_profile_info.ready_time;
      }
    }
    actor_profile_info.set_act_num(act_num);
    actor_profile_info.set_avg_act_time(acc_act_time / act_num);
    if (act_num > 1) {
      actor_profile_info.set_avg_act_interval(acc_act_interval / (act_num - 1));
      CHECK(experiment_actor_id2profile_info.insert({pair.first, actor_profile_info}).second);
    }
  }

  auto act_events = of_make_unique<std::list<ActEvent>>();
  ParseActEvents(act_event_filepath, act_events.get());
  HashMap<int64_t, std::vector<ActProfileInfo>> actor_id2act_profile_info;
  for (auto& act_event : *act_events.get()) {
    int64_t actor_id = act_event.actor_id();
    ActProfileInfo act_profile_info(
        {act_event.ready_time(), act_event.start_time(), act_event.stop_time()});
    std::vector<ActProfileInfo>& act_profile_infos = actor_id2act_profile_info[actor_id];
    act_profile_infos.emplace_back(act_profile_info);
  }

  using ProfileInfoPair = std::pair<int64_t, ActorProfileInfo>;
  std::vector<ProfileInfoPair> profile_info_vec;
  for (auto& pair : actor_id2act_profile_info) {
    std::vector<ActProfileInfo>& act_profile_infos = pair.second;
    std::sort(act_profile_infos.begin(), act_profile_infos.end(),
              [](const ActProfileInfo& lhs, const ActProfileInfo& rhs) {
                return lhs.ready_time < rhs.ready_time;
              });
    ProfileInfoPair profile_info_pair;
    profile_info_pair.first = pair.first;
    int64_t act_num = act_profile_infos.size();
    double acc_act_time = 0;
    double last_ready_time = -1;
    double acc_act_interval = 0;
    for (auto& act_profile_info : act_profile_infos) {
      acc_act_time += (act_profile_info.stop_time - act_profile_info.start_time);
      if (last_ready_time < 0) {
        last_ready_time = act_profile_info.ready_time;
      } else {
        acc_act_interval += (act_profile_info.ready_time - last_ready_time);
        last_ready_time = act_profile_info.ready_time;
      }
    }
    profile_info_pair.second.set_act_num(act_num);
    if (Global<JobDesc>::Get()->record_nonexperiment_level() == 1) {
      auto it = experiment_actor_id2profile_info.find(pair.first);
      if (it == experiment_actor_id2profile_info.end()) {
        profile_info_pair.second.set_avg_act_time(0);
      } else {
        double experiment_avg_act_time = it->second.avg_act_time();
        profile_info_pair.second.set_avg_act_time(experiment_avg_act_time);
      }
    } else {
      profile_info_pair.second.set_avg_act_time(acc_act_time / act_num);
    }
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
