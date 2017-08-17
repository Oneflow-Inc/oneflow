#ifndef ONEFLOW_CORE_COMMON_RUNTIME_RUNTIME_H_
#define ONEFLOW_CORE_COMMON_RUNTIME_RUNTIME_H_
#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

namespace runtime {

class Runtime final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Runtime);
  ~Runtime() = default;
  OF_SINGLETON(Runtime);
  void Run(const Plan& plan, const std::string& this_machine_name);

  void SetPlan(const Plan& plan);
  void SetThisMachineName(const std::string& this_machine_name);

  void InitRuntime();
  void InitModel();
  void ActivateActor();
  void SendRemoteRegstToInc();
  void SendRemoteRegstToDec();
  void StartActor();

 private:
  Runtime() = default;
  void InitSingleton(const Plan& plan, const std::string& this_machine_name);
  void DeleteSingleton();
  void HandoutTasks(const std::vector<const TaskProto*>& tasks);
  void SendCmdMsg(const std::vector<const TaskProto*>& tasks, ActorCmd cmd);

  void FindTasksOnThisMachine();

  Plan plan_;
  std::string this_machine_name_;

  std::vector<const TaskProto*> mdupdt_tasks_;
  std::vector<const TaskProto*> source_tasks_;
  std::vector<const TaskProto*> other_tasks_;
  size_t this_machine_task_num_;
};
}  // namespace runtime
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_RUNTIME_RUNTIME_H_
