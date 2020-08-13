/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/comm_network/epoll/epoll_comm_network.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/runtime_job_descs.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/actor/act_event_logger.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/customized/summary/events_writer.h"
#include "oneflow/core/job/collective_boxing_executor.h"
#include "oneflow/core/job/collective_boxing_device_ctx_poller.h"

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
  LOG(INFO) << "Actors on this machine constructed";
  OF_BARRIER();
  LOG(INFO) << "Actors on every machine constructed";
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
  if (Global<ResourceDesc, ForSession>::Get()->TotalMachineNum() > 1) {
#ifdef PLATFORM_POSIX
    if (Global<ResourceDesc, ForSession>::Get()->use_rdma()) {
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
  Global<boxing::collective::CollectiveBoxingExecutor>::New(plan);
  Global<MemoryAllocator>::New();
  Global<RegstMgr>::New(plan);
  Global<ActorMsgBus>::New();
  Global<ThreadMgr>::New(plan);
  Global<boxing::collective::CollectiveBoxingDeviceCtxPoller>::New();
  Global<RuntimeJobDescs>::New(plan.job_confs().job_id2job_conf());
  Global<summary::EventsWriter>::New();
}

void Runtime::DeleteAllGlobal() {
  Global<RuntimeJobDescs>::Delete();
  Global<boxing::collective::CollectiveBoxingDeviceCtxPoller>::Delete();
  Global<ThreadMgr>::Delete();
  Global<ActorMsgBus>::Delete();
  Global<RegstMgr>::Delete();
  Global<MemoryAllocator>::Delete();
  Global<boxing::collective::CollectiveBoxingExecutor>::Delete();
  Global<CommNet>::Delete();
  Global<ActEventLogger>::Delete();
  Global<RuntimeCtx>::Delete();
  Global<summary::EventsWriter>::Delete();
}

}  // namespace oneflow
