#include "oneflow/core/job/runtime.h"
#include "oneflow/core/comm_network/epoll/epoll_comm_network.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/actor/act_event_logger.h"

namespace oneflow {

namespace {

void SendCmdMsg(const std::vector<const TaskProto*>& tasks, ActorCmd cmd) {
  for (const TaskProto* task : tasks) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(task->task_id(), cmd);
    ActorMsgBus::Singleton()->SendMsg(msg);
  }
}

void HandoutTasks(const std::vector<const TaskProto*>& tasks) {
  for (const TaskProto* task : tasks) {
    ThreadMgr::Singleton()->GetThrd(task->thrd_id())->AddTask(*task);
  }
  SendCmdMsg(tasks, ActorCmd::kConstructActor);
}

}  // namespace

Runtime::Runtime(const Plan& plan, bool is_experiment_phase) {
  NewAllSingleton(plan, is_experiment_phase);
  std::vector<const TaskProto*> mdupdt_tasks;
  std::vector<const TaskProto*> source_tasks;
  std::vector<const TaskProto*> other_tasks;
  int64_t this_machine_task_num = 0;
  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != MachineCtx::Singleton()->this_machine_id()) {
      continue;
    }
    if (task.task_type() == TaskType::kMdUpdt) {
      mdupdt_tasks.push_back(&task);
    } else if (task.task_type() == TaskType::kSource) {
      source_tasks.push_back(&task);
    } else {
      other_tasks.push_back(&task);
    }
    this_machine_task_num += 1;
  }
  RuntimeCtx* runtime_ctx = RuntimeCtx::Singleton();
  runtime_ctx->NewCounter("constructing_actor_cnt", this_machine_task_num);
  HandoutTasks(mdupdt_tasks);
  HandoutTasks(source_tasks);
  HandoutTasks(other_tasks);
  runtime_ctx->WaitUntilCntEqualZero("constructing_actor_cnt");
  LOG(INFO) << "All actor on this machine are constructed";
  OF_BARRIER();
  LOG(INFO) << "All actor on all machine are constructed";
  CommNet::Singleton()->RegisterMemoryDone();
  runtime_ctx->NewCounter("model_init_cnt", mdupdt_tasks.size());
  SendCmdMsg(mdupdt_tasks, ActorCmd::kInitModel);
  runtime_ctx->WaitUntilCntEqualZero("model_init_cnt");
  LOG(INFO) << "InitModel on this machine done";
  OF_BARRIER();
  LOG(INFO) << "InitModel on all machine done";
  runtime_ctx->NewCounter("running_actor_cnt", this_machine_task_num);
  SendCmdMsg(mdupdt_tasks, ActorCmd::kSendInitialModel);
  SendCmdMsg(source_tasks, ActorCmd::kStart);
  runtime_ctx->WaitUntilCntEqualZero("running_actor_cnt");
  OF_BARRIER();
  DeleteAllSingleton();
}

void Runtime::NewAllSingleton(const Plan& plan, bool is_experiment_phase) {
  const JobDesc* job_desc = JobDesc::Singleton();
  int64_t piece_num = 0;
  if (is_experiment_phase) {
    piece_num = job_desc->piece_num_of_experiment_phase();
    ActEventLogger::NewSingleton();
  } else {
    if (job_desc->IsTrain()) {
      piece_num = job_desc->NumOfPiecesInBatch() * job_desc->TotalBatchNum();
    } else {
      piece_num = std::numeric_limits<int64_t>::max();
    }
  }
  RuntimeCtx::NewSingleton(piece_num, is_experiment_phase);
#ifdef PLATFORM_POSIX
  if (JobDesc::Singleton()->use_rdma()) {
#ifdef WITH_RDMA
    IBVerbsCommNet::Init();
#else
    EpollCommNet::Init();
#endif
  } else {
    EpollCommNet::Init();
  }
#endif
  SnapshotMgr::NewSingleton(plan);
  MemoryAllocator::NewSingleton();
  RegstMgr::NewSingleton();
  ActorMsgBus::NewSingleton();
  ThreadMgr::NewSingleton();
}

void Runtime::DeleteAllSingleton() {
  ThreadMgr::DeleteSingleton();
  ActorMsgBus::DeleteSingleton();
  RegstMgr::DeleteSingleton();
  MemoryAllocator::DeleteSingleton();
  SnapshotMgr::DeleteSingleton();
  delete CommNet::Singleton();
  RuntimeCtx::DeleteSingleton();
  ActEventLogger::DeleteSingleton();
}

}  // namespace oneflow
