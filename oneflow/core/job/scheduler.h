#ifndef ONEFLOW_CORE_JOB_SCHEDULER_H_
#define ONEFLOW_CORE_JOB_SCHEDULER_H_

#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class Scheduler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Scheduler);
  ~Scheduler() = default;

  OF_SINGLETON(Scheduler);

  void Process(const JobConf& job_conf, const std::string& this_machine_name);

 private:
  Scheduler() = default;
  Plan GetPlanFromJobConf(const JobConf& job_conf,
                          const std::string& this_machine_name);
  void DeleteAllSingleton();
  void HandoutTasks(const std::vector<const TaskProto*>& tasks);
  void SendCmdMsg(const std::vector<const TaskProto*>& tasks, ActorCmd cmd);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SCHEDULER_H_
