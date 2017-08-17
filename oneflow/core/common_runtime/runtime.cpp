#include "oneflow/core/common_runtime/runtime.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

namespace runtime {

void Runtime::SetPlan(const Plan& plan) { plan_ = plan; }

void Runtime::SetThisMachineName(const std::string& this_machine_name) {
  this_machine_name_ = this_machine_name;
}

void Runtime::InitRuntime() {
  InitSingleton(plan_, this_machine_name_);
  FindTasksOnThisMachine();
}

void Runtime::InitModel() {
  RuntimeCtx::Singleton()->mut_model_init_cnt().Init("model_init_cnt",
                                                     mdupdt_tasks_.size());
  HandoutTasks(mdupdt_tasks_);
  SendCmdMsg(mdupdt_tasks_, ActorCmd::kInitializeModel);
  RuntimeCtx::Singleton()->mut_model_init_cnt().WaitUntilCntEqualZero();
}

void Runtime::ActivateActor() {
  RuntimeCtx::Singleton()->mut_inactive_actor_cnt().Init(
      "inactive_actor_cnt", this_machine_task_num_ - mdupdt_tasks_.size());
  HandoutTasks(source_tasks_);
  HandoutTasks(other_tasks_);
  RuntimeCtx::Singleton()->mut_inactive_actor_cnt().WaitUntilCntEqualZero();
  LOG(INFO) << "All actor on this machine are activated";
  // OF_BARRIER();
  LOG(INFO) << "All actor on all machine are activated";
}

void Runtime::SendRemoteRegstToInc() {}

void Runtime::SendRemoteRegstToDec() {}

void Runtime::StartActor() {}

void Runtime::FindTasksOnThisMachine() {
  for (const TaskProto& task : plan_.task()) {
    if (task.machine_id() != RuntimeCtx::Singleton()->this_machine_id()) {
      continue;
    }
    if (task.type() == kMdUpdtCompTask) {
      mdupdt_tasks_.push_back(&task);
    } else if (task.consumed_regst_desc_id().empty()) {
      source_tasks_.push_back(&task);
    } else {
      other_tasks_.push_back(&task);
    }
  }

  this_machine_task_num_ =
      mdupdt_tasks_.size() + source_tasks_.size() + other_tasks_.size();
  LOG(INFO) << "number of mdupdt tasks is " << mdupdt_tasks_.size();
  LOG(INFO) << "number of source tasks is " << source_tasks_.size();
  LOG(INFO) << "number of other  tasks is " << other_tasks_.size();
}

void Runtime::Run(const Plan& plan, const std::string& this_machine_name) {
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
  // OF_BARRIER();
  LOG(INFO) << "InitModel on all machine done";
  HandoutTasks(source_tasks);
  HandoutTasks(other_tasks);
  RuntimeCtx::Singleton()->mut_inactive_actor_cnt().WaitUntilCntEqualZero();
  LOG(INFO) << "All actor on this machine are activated";
  // OF_BARRIER();
  LOG(INFO) << "All actor on all machine are activated";
  // Network Swap Memory Message
  RuntimeCtx::Singleton()->mut_active_actor_cnt().Init("active_actor_cnt",
                                                       this_machine_task_num);
  SendCmdMsg(mdupdt_tasks, ActorCmd::kSendInitialModel);
  SendCmdMsg(source_tasks, ActorCmd::kStart);
  RuntimeCtx::Singleton()->mut_active_actor_cnt().WaitUntilCntEqualZero();
  DeleteSingleton();
}

void Runtime::InitSingleton(const Plan& plan,
                            const std::string& this_machine_name) {
  JobDesc::Singleton()->InitFromProto(plan.job_desc());
  IDMgr::Singleton()->InitFromResource(JobDesc::Singleton()->resource());
  RuntimeCtx::Singleton()->set_this_machine_name(this_machine_name);
  KernelMgr::Singleton()->InitFromPlan(plan);
  SnapshotMgr::Singleton()->Init();
  ActorMsgBus::Singleton()->Init();
  ThreadMgr::Singleton();
}
void Runtime::DeleteSingleton() {
  delete ThreadMgr::Singleton();
  delete ActorMsgBus::Singleton();
  delete SnapshotMgr::Singleton();
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
    ActorMsg msg;
    msg.set_dst_actor_id(IDMgr::Singleton()->ActorId4TaskId(task->id()));
    msg.set_actor_cmd(cmd);
    ActorMsgBus::Singleton()->SendMsg(msg);
  }
}

}  // namespace runtime
}  // namespace oneflow
