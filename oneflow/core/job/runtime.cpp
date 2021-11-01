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
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/job/runtime_job_descs.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/user/summary/events_writer.h"

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

Runtime::Runtime(const Plan& plan, const HashMap<std::string, Blob*>& variable_op_name2eager_blob) {
  {
    // NOTE(chengcheng): All runtime Global objects AddPlan
    Global<RegstMgr>::Get()->AddPlan(plan, variable_op_name2eager_blob);
    Global<ThreadMgr>::Get()->AddPlan(plan);
    Global<RuntimeJobDescs>::Get()->AddPlan(plan);
    collective_boxing_scheduler_plan_token_ =
        Global<boxing::collective::Scheduler>::Get()->AddPlan(plan);
  }
  std::vector<const TaskProto*> source_tasks;
  std::vector<const TaskProto*> other_tasks;
  int64_t this_machine_task_num = 0;
  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != GlobalProcessCtx::Rank()) { continue; }
    if (!HasNonCtrlConsumedRegstDescId(task)) {
      source_tasks.push_back(&task);
    } else {
      other_tasks.push_back(&task);
    }
    auto it = job_id2actor_size_.find(task.job_id());
    if (it == job_id2actor_size_.end()) {
      auto emplace_ret_pair = job_id2actor_size_.emplace(task.job_id(), 0);
      CHECK(emplace_ret_pair.second);
      it = emplace_ret_pair.first;
    }
    it->second++;
    this_machine_task_num++;
  }
  RuntimeCtx* runtime_ctx = Global<RuntimeCtx>::Get();
  runtime_ctx->NewCounter("constructing_actor_cnt", this_machine_task_num);
  HandoutTasks(source_tasks);
  HandoutTasks(other_tasks);
  runtime_ctx->WaitUntilCntEqualZero("constructing_actor_cnt");
  LOG(INFO) << "Actors on this machine constructed";
  OF_SESSION_BARRIER();
  LOG(INFO) << "Actors on every machine constructed";
  for (auto pair : job_id2actor_size_) {
    runtime_ctx->NewCounter(GetRunningActorCountKeyByJobId(pair.first), pair.second);
  }
  SendCmdMsg(source_tasks, ActorCmd::kStart);
}

Runtime::~Runtime() {
  for (auto pair : job_id2actor_size_) {
    Global<RuntimeCtx>::Get()->WaitUntilCntEqualZero(GetRunningActorCountKeyByJobId(pair.first));
  }
  OF_SESSION_BARRIER();
  Global<boxing::collective::Scheduler>::Get()->DeletePlan(collective_boxing_scheduler_plan_token_);
}

}  // namespace oneflow
