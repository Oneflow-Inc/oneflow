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

 private:
  Runtime() = default;
  void InitSingleton(const Plan& plan, const std::string& this_machine_name);
  void DeleteSingleton();
  void HandoutTasks(const std::vector<const TaskProto*>& tasks);
  void SendCmdMsg(const std::vector<const TaskProto*>& tasks, ActorCmd cmd);
};
}  // namespace runtime
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_RUNTIME_RUNTIME_H_
