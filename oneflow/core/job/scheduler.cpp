#include "oneflow/core/job/scheduler.h"
#include "oneflow/core/comm_network/rdma_data_comm_network.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

void Scheduler::Process(const JobConf& job_conf,
                        const std::string& this_machine_name) {
  Plan plan = GetPlanFromJobConf(job_conf, this_machine_name);
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
  DataCommNet::Singleton()->RegisterMemoryDone();
  RuntimeCtx::Singleton()->mut_active_actor_cnt().Init("active_actor_cnt",
                                                       this_machine_task_num);
  SendCmdMsg(mdupdt_tasks, ActorCmd::kSendInitialModel);
  SendCmdMsg(source_tasks, ActorCmd::kStart);
  RuntimeCtx::Singleton()->mut_active_actor_cnt().WaitUntilCntEqualZero();
  DeleteAllSingleton();
}

Plan Scheduler::GetPlanFromJobConf(const JobConf& job_conf,
                                   const std::string& this_machine_name) {
  JobDesc::Singleton()->InitFromJobConf(job_conf);
  IDMgr::Singleton()->Init();
  RuntimeCtx::Singleton()->set_this_machine_name(this_machine_name);
  CtrlCommNet::Singleton()->Init();
  Plan plan;
  if (RuntimeCtx::Singleton()->IsThisMachineMaster()) {
    plan = Compiler::Singleton()->Compile();
    OpMgr::RefreshSingleton();
    // TODO: send plan
  } else {
    // TODO: receive plan
  }
  KernelMgr::Singleton()->InitFromPlan(plan);
  RdmaDataCommNet::Init();
  SnapshotMgr::Singleton()->Init(plan);
  ActorMsgBus::Singleton()->Init();
  ThreadMgr::Singleton();
  return plan;
}
void Scheduler::DeleteAllSingleton() {
  delete ThreadMgr::Singleton();
  delete ActorMsgBus::Singleton();
  delete SnapshotMgr::Singleton();
}
void Scheduler::HandoutTasks(const std::vector<const TaskProto*>& tasks) {
  for (const TaskProto* task : tasks) {
    ThreadMgr::Singleton()->GetThrd(task->thrd_local_id())->AddTask(*task);
  }
  SendCmdMsg(tasks, ActorCmd::kActivateActor);
}
void Scheduler::SendCmdMsg(const std::vector<const TaskProto*>& tasks,
                           ActorCmd cmd) {
  for (const TaskProto* task : tasks) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(
        IDMgr::Singleton()->ActorId4TaskId(task->id()), cmd);
    ;
    ActorMsgBus::Singleton()->SendMsg(msg);
  }
}

}  // namespace oneflow

DEFINE_string(job_conf, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::JobConf job_conf;
  oneflow::ParseProtoFromTextFile(FLAGS_job_conf, &job_conf);
  oneflow::Scheduler::Singleton()->Process(job_conf, FLAGS_this_machine_name);
  return 0;
}
