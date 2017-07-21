#include "gflags/gflags.h"
#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

class Runtime final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Runtime);
  ~Runtime() = default;

  OF_SINGLETON(Runtime);

  void Run(const Plan& plan, const std::string& this_machine_name) {
    InitSingleton(plan, this_machine_name);
    std::vector<const TaskProto*> mdupdt_tasks;
    std::vector<const TaskProto*> source_tasks;
    std::vector<const TaskProto*> other_tasks;
    for (const TaskProto& task : plan.task()) {
      if (task.machine_id() != RuntimeCtx::Singleton()->this_machine_id()) {
        continue;
      }
      if (task.type() == kMdUpdtCompTask) {
        mdupdt_tasks.push_back(&task);
      } else if (task.subscribed_regst_desc_id().empty()) {
        source_tasks.push_back(&task);
      } else {
        other_tasks.push_back(&task);
      }
    }
    LOG(INFO) << "number of mdupdt tasks is " << mdupdt_tasks.size();
    LOG(INFO) << "number of source tasks is " << source_tasks.size();
    LOG(INFO) << "number of other  tasks is " << other_tasks.size();
    HandoutTasks(mdupdt_tasks);
    RuntimeCtx::Singleton()->SetModelInitCnt(mdupdt_tasks.size());
    SendCmdMsg(mdupdt_tasks, ActorCmd::kInitializeModel);
    HandoutTasks(source_tasks);
    HandoutTasks(other_tasks);
    RuntimeCtx::Singleton()->WaitUnitlAllModelInitDone();
    LOG(INFO) << "InitModel on this machine done";
    // OF_BARRIER();
    LOG(INFO) << "InitModel on all machine done";
    SendCmdMsg(mdupdt_tasks, ActorCmd::kSendInitialModel);
    SendCmdMsg(source_tasks, ActorCmd::kStart);
    DeleteSingleton();
  }

 private:
  Runtime() = default;
  void InitSingleton(const Plan& plan, const std::string& this_machine_name) {
    JobDesc::Singleton()->InitFromProto(plan.job_desc());
    IDMgr::Singleton()->InitFromResource(JobDesc::Singleton()->resource());
    RuntimeCtx::Singleton()->set_this_machine_name(this_machine_name);
    KernelMgr::Singleton()->InitFromPlan(plan);
    SnapshotMgr::Singleton()->Init();
    ActorMsgBus::Singleton()->Init();
    ThreadMgr::Singleton();
  }
  void DeleteSingleton() {
    delete ThreadMgr::Singleton();
    delete ActorMsgBus::Singleton();
    delete SnapshotMgr::Singleton();
  }
  void HandoutTasks(const std::vector<const TaskProto*>& tasks) {
    for (const TaskProto* task : tasks) {
      ThreadMgr::Singleton()->GetThrd(task->thrd_local_id())->AddTask(*task);
    }
  }
  void SendCmdMsg(const std::vector<const TaskProto*>& tasks, ActorCmd cmd) {
    for (const TaskProto* task : tasks) {
      ActorMsg msg;
      msg.set_dst_actor_id(IDMgr::Singleton()->ActorId4TaskId(task->id()));
      msg.set_actor_cmd(cmd);
      ActorMsgBus::Singleton()->SendMsg(msg);
    }
  }
};

}  // namespace oneflow

DEFINE_string(plan_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Runtime Starting Up";
  oneflow::Plan plan;
  LOG(INFO) << "Parse Plan File";
  oneflow::ParseProtoFromTextFile(FLAGS_plan_filepath, &plan);
  oneflow::Runtime::Singleton()->Run(plan, FLAGS_this_machine_name);
  LOG(INFO) << "Runtime Shutting Down";
  return 0;
}
