#include "oneflow/core/job/runtime.h"
#include "oneflow/core/comm_network/epoll/epoll_comm_network.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/runtime_job_descs.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/actor/act_event_logger.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace {

void SendCmdMsg(const std::vector<const TaskProto*>& tasks, ActorCmd cmd) {
  for (const TaskProto* task : tasks) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(task->task_id(), cmd);
    Global<ActorMsgBus>::Get()->SendMsg(msg);
  }
}

void HandoutTasks(const std::vector<const TaskProto*>& tasks) {
  for (const TaskProto* task : tasks) {
    Global<ThreadMgr>::Get()->GetThrd(task->thrd_id())->AddTask(*task);
  }
  SendCmdMsg(tasks, ActorCmd::kConstructActor);
}

bool HasNonCtrlConsumedRegstDescId(const TaskProto& task) {
  for (const auto& pair : task.consumed_regst_desc_id()) {
    if (pair.first == "in_ctrl") { continue; }
    return true;
  }
  return false;
}

}  // namespace

Runtime::Runtime(const Plan& plan, size_t total_piece_num, bool is_experiment_phase) {
  NewAllGlobal(plan, total_piece_num, is_experiment_phase);
  std::vector<const TaskProto*> mdupdt_tasks;
  std::vector<const TaskProto*> source_tasks;
  std::vector<const TaskProto*> other_tasks;
  int64_t this_machine_task_num = 0;
  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != Global<MachineCtx>::Get()->this_machine_id()) { continue; }
    if (IsMdUpdtTaskType(task.task_type())) {
      mdupdt_tasks.push_back(&task);
    } else if (!HasNonCtrlConsumedRegstDescId(task)) {
      source_tasks.push_back(&task);
    } else {
      other_tasks.push_back(&task);
    }
    this_machine_task_num += 1;
  }
  RuntimeCtx* runtime_ctx = Global<RuntimeCtx>::Get();
  runtime_ctx->NewCounter("constructing_actor_cnt", this_machine_task_num);
  HandoutTasks(mdupdt_tasks);
  HandoutTasks(source_tasks);
  HandoutTasks(other_tasks);
  runtime_ctx->WaitUntilCntEqualZero("constructing_actor_cnt");
  LOG(INFO) << "All actor on this machine are constructed";
  OF_BARRIER();
  LOG(INFO) << "All actor on all machine are constructed";
  if (Global<CommNet>::Get()) { Global<CommNet>::Get()->RegisterMemoryDone(); }
  runtime_ctx->NewCounter("model_init_cnt", mdupdt_tasks.size());
  SendCmdMsg(mdupdt_tasks, ActorCmd::kInitModel);
  runtime_ctx->WaitUntilCntEqualZero("model_init_cnt");
  LOG(INFO) << "InitModel on this machine done";
  OF_BARRIER();
  LOG(INFO) << "InitModel on all machine done";
  runtime_ctx->NewCounter("running_actor_cnt", this_machine_task_num);
  SendCmdMsg(mdupdt_tasks, ActorCmd::kSendInitialModel);
  SendCmdMsg(source_tasks, ActorCmd::kStart);
}

Runtime::~Runtime() {
  Global<RuntimeCtx>::Get()->WaitUntilCntEqualZero("running_actor_cnt");
  OF_BARRIER();
  DeleteAllGlobal();
}

void Runtime::NewAllGlobal(const Plan& plan, size_t total_piece_num, bool is_experiment_phase) {
  Global<RuntimeCtx>::New(total_piece_num, is_experiment_phase);
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()
      && Global<RuntimeCtx>::Get()->NeedCollectActEvent()) {
    Global<ActEventLogger>::New(is_experiment_phase);
  }
  if (Global<ResourceDesc>::Get()->TotalMachineNum() > 1) {
#ifdef PLATFORM_POSIX
    if (Global<ResourceDesc>::Get()->use_rdma()) {
#ifdef WITH_RDMA
      IBVerbsCommNet::Init(plan);
#else
      LOG(FATAL) << "RDMA components not found";
#endif
    } else {
      EpollCommNet::Init(plan);
    }
#endif
  }
#ifdef WITH_CUDA
  InitGlobalCudaDeviceProp();
#endif
  Global<MemoryAllocator>::New();
  Global<RegstMgr>::New(plan);
  Global<ActorMsgBus>::New();
  Global<ThreadMgr>::New(plan);
#ifdef WITH_CUDA
  Global<NcclCommMgr>::New(plan);
#endif  // WITH_CUDA
  Global<RuntimeJobDescs>::New(plan.job_confs().job_id2job_conf());
}

void Runtime::DeleteAllGlobal() {
  Global<RuntimeJobDescs>::Delete();
#ifdef WITH_CUDA
  Global<NcclCommMgr>::Delete();
#endif  // WITH_CUDA
  Global<ThreadMgr>::Delete();
  Global<ActorMsgBus>::Delete();
  Global<RegstMgr>::Delete();
  Global<MemoryAllocator>::Delete();
  Global<CommNet>::Delete();
  Global<ActEventLogger>::Delete();
  Global<RuntimeCtx>::Delete();
}

}  // namespace oneflow
