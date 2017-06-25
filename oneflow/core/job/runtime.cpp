#include "gflags/gflags.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/runtime_info.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/actor/actor_message_bus.h"

namespace oneflow {

class Runtime final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Runtime);
  ~Runtime() = default;

  OF_SINGLETON(Runtime);

  void Run(const Plan& plan, const std::string& this_machine_name) {
    InitSingleton(plan, this_machine_name);
    AddMdUpdtCompTaskAndInitModel(plan);
    AddTheOtherTasks(plan);
    SendInitialModel(plan);
    // send msg to source actor ?
  }

 private:
  Runtime() = default;
  void InitSingleton(const Plan& plan, const std::string& this_machine_name) {
    JobDesc::Singleton().InitFromProto(plan.job_desc());
    IDMgr::Singleton().InitFromResource(JobDesc::Singleton().resource());
    RuntimeInfo::Singleton().set_this_machine_name(this_machine_name);
    KernelMgr::Singleton().InitFromPlan(plan);
  }
  void AddMdUpdtCompTaskAndInitModel(const Plan& plan) {
    for (const TaskProto& task : plan.task()) {
      if (task.type() == kMdUpdtCompTask) {
        ThreadMgr::Singleton().GetThrd(task.thrd_local_id())->AddTask(task);
        ActorMsg msg;
        msg.set_dst_actor_id(IDMgr::Singleton().ActorId4TaskId(task.id()));
        msg.set_actor_cmd(ActorCmd::kInitializeModel);
        ActorMsgBus::Singleton().SendMsg(msg);
      }
    }
  }
  void AddTheOtherTasks(const Plan& plan) {
    for (const TaskProto& task : plan.task()) {
      if (task.type() != kMdUpdtCompTask) {
        ThreadMgr::Singleton().GetThrd(task.thrd_local_id())->AddTask(task);
      }
    }
  }
  void SendInitialModel(const Plan& plan) {
    for (const TaskProto& task : plan.task()) {
      if (task.type() == kMdUpdtCompTask) {
        ActorMsg msg;
        msg.set_dst_actor_id(IDMgr::Singleton().ActorId4TaskId(task.id()));
        msg.set_actor_cmd(ActorCmd::kSendInitialModel);
        ActorMsgBus::Singleton().SendMsg(msg);
      }
    }
  }

};

} // namespace oneflow

DEFINE_string(plan_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Runtime Starting Up...";
  oneflow::Plan plan;
  oneflow::ParseProtoFromTextFile(FLAGS_plan_filepath, &plan);
  oneflow::Runtime::Singleton().Run(plan, FLAGS_this_machine_name);
  LOG(INFO) << "Runtime Shutting Down...";
  return 0;
}
