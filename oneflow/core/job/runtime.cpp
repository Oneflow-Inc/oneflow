#include <gflags/gflags.h>
#include "oneflow/core/comm_network/epoll/epoll_comm_network.h"
#include "oneflow/core/comm_network/rdma/rdma_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

class Runtime final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Runtime);
  ~Runtime() = default;

  OF_SINGLETON(Runtime);

  void Run(const Plan& plan, const std::string& this_machine_name);

 private:
  Runtime() = default;
  void NewAllSingleton(const Plan& plan, const std::string& this_machine_name);
  void DeleteAllSingleton();
  void HandoutTasks(const std::vector<const TaskProto*>& tasks);
  void SendCmdMsg(const std::vector<const TaskProto*>& tasks, ActorCmd cmd);
};

void Runtime::Run(const Plan& plan, const std::string& this_machine_name) {
  NewAllSingleton(plan, this_machine_name);
  CommNet::Singleton()->EstablishNetwork();
  // find tasks on this machine
  std::vector<const TaskProto*> mdupdt_tasks;
  std::vector<const TaskProto*> source_tasks;
  std::vector<const TaskProto*> other_tasks;
  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != RuntimeCtx::Singleton()->this_machine_id()) {
      continue;
    }
    if (task.type() == kMdUpdtCompTask) {
      mdupdt_tasks.push_back(&task);
    } else if (task.consumed_regst_desc_id().empty()) {
      source_tasks.push_back(&task);
    } else {
      other_tasks.push_back(&task);
    }
  }
  size_t this_machine_task_num =
      mdupdt_tasks.size() + source_tasks.size() + other_tasks.size();
  LOG(INFO) << "number of mdupdt tasks is " << mdupdt_tasks.size();
  LOG(INFO) << "number of source tasks is " << source_tasks.size();
  LOG(INFO) << "number of other  tasks is " << other_tasks.size();
  RuntimeCtx::Singleton()->mut_inactive_actor_cnt().Init("inactive_actor_cnt",
                                                         this_machine_task_num);
  RuntimeCtx::Singleton()->mut_model_init_cnt().Init("model_init_cnt",
                                                     mdupdt_tasks.size());
  HandoutTasks(mdupdt_tasks);
  SendCmdMsg(mdupdt_tasks, ActorCmd::kInitializeModel);
  RuntimeCtx::Singleton()->mut_model_init_cnt().WaitUntilCntEqualZero();
  LOG(INFO) << "InitModel on this machine done";
  OF_BARRIER();
  LOG(INFO) << "InitModel on all machine done";
  HandoutTasks(source_tasks);
  HandoutTasks(other_tasks);
  RuntimeCtx::Singleton()->mut_inactive_actor_cnt().WaitUntilCntEqualZero();
  LOG(INFO) << "All actor on this machine are activated";
  OF_BARRIER();
  LOG(INFO) << "All actor on all machine are activated";
  CommNet::Singleton()->RegisterMemoryDone();
  RuntimeCtx::Singleton()->mut_active_actor_cnt().Init("active_actor_cnt",
                                                       this_machine_task_num);
  SendCmdMsg(mdupdt_tasks, ActorCmd::kSendInitialModel);
  SendCmdMsg(source_tasks, ActorCmd::kStart);
  RuntimeCtx::Singleton()->mut_active_actor_cnt().WaitUntilCntEqualZero();
  OF_BARRIER();
  DeleteAllSingleton();
}

void Runtime::NewAllSingleton(const Plan& plan,
                              const std::string& this_machine_name) {
  JobDesc::NewSingleton(plan.job_desc());
  IDMgr::NewSingleton();
  RuntimeCtx::NewSingleton(this_machine_name);
  CtrlClient::NewSingleton();
  KernelMgr::NewSingleton(plan);
#ifdef PLATFORM_POSIX
  if (JobDesc::Singleton()->use_rdma()) {
    RdmaCommNet::Init();
  } else {
    EpollCommNet::Init();
  }
#endif
  SnapshotMgr::NewSingleton(plan);
  RegstMgr::NewSingleton();
  ActorMsgBus::NewSingleton();
  ThreadMgr::NewSingleton();
}

void Runtime::DeleteAllSingleton() {
  ThreadMgr::DeleteSingleton();
  ActorMsgBus::DeleteSingleton();
  RegstMgr::DeleteSingleton();
  SnapshotMgr::DeleteSingleton();
  delete CommNet::Singleton();
  KernelMgr::DeleteSingleton();
  CtrlClient::DeleteSingleton();
  RuntimeCtx::DeleteSingleton();
  IDMgr::DeleteSingleton();
  JobDesc::DeleteSingleton();
}
void Runtime::HandoutTasks(const std::vector<const TaskProto*>& tasks) {
  for (const TaskProto* task : tasks) {
    ThreadMgr::Singleton()->GetThrd(task->thrd_local_id())->AddTask(*task);
  }
  SendCmdMsg(tasks, ActorCmd::kActivateActor);
}
void Runtime::SendCmdMsg(const std::vector<const TaskProto*>& tasks,
                         ActorCmd cmd) {
  for (const TaskProto* task : tasks) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(
        IDMgr::Singleton()->ActorId4TaskId(task->id()), cmd);
    ActorMsgBus::Singleton()->SendMsg(msg);
  }
}

}  // namespace oneflow

DEFINE_string(plan_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::RedirectStdoutAndStderrToGlogDir();
  LOG(INFO) << "Runtime Start";
  oneflow::Plan plan;
  oneflow::ParseProtoFromTextFile(FLAGS_plan_filepath, &plan);
  oneflow::Runtime::NewSingleton();
  oneflow::Runtime::Singleton()->Run(plan, FLAGS_this_machine_name);
  oneflow::Runtime::DeleteSingleton();
  oneflow::CloseStdoutAndStderr();
  LOG(INFO) << "Runtime Stop";
  return 0;
}
