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
#include "oneflow/core/job/cluster.h"
#include "oneflow/core/job/cluster_instruction.pb.h"
#include "oneflow/core/job/cluster_instruction.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/thread/thread_pool.h"

namespace oneflow {

namespace {

void AsyncRunLazyJobSet(ThreadPool* lazy_runtime_thread,
                        std::shared_ptr<BlockingCounter> wait_session_init) {
  lazy_runtime_thread->AddWork([wait_session_init] {
    ConfigProto config_proto;
    Singleton<CtrlClient>::Get()->PullKV("config_proto", &config_proto);
    CHECK_NOTNULL(Singleton<EnvDesc>::Get());
    int32_t machine_num = Singleton<EnvDesc>::Get()->TotalMachineNum();
    // do nothing if it's not my business
    if (GlobalProcessCtx::Rank() >= machine_num) { return; }
    Singleton<SessionGlobalObjectsScope>::New();
    CHECK_JUST(Singleton<SessionGlobalObjectsScope>::Get()->Init(config_proto));
    wait_session_init->Decrease();
    JobSet job_set;
    Singleton<CtrlClient>::Get()->PullKV("session_job_set", &job_set);
    {
      Oneflow oneflow;
      CHECK_JUST(oneflow.Init(job_set));
    }
    Singleton<SessionGlobalObjectsScope>::Delete();
  });
}

}  // namespace

Maybe<void> Cluster::WorkerLoop() {
  // The reason why excluding master machine is that
  // eager instruction for compile-time symbol constructing must be done synchronously
  CHECK_OR_RETURN(!GlobalProcessCtx::IsThisProcessMaster());
  {
    // Oneflow::~Oneflow blocking in current thread is not acceptable
    // Two reasons why `lazy_runtime_thread` is needed:
    //   1. making current thread non-block by
    //      taking over the execution of Oneflow::~Oneflow
    //   2. as a Synchronizing guard for all unfinished Oneflow::~Oneflow
    //
    // thread_num must be 1.
    ThreadPool lazy_runtime_thread(1);
    std::list<std::shared_ptr<BlockingCounter>> wait_session_init_list;
    while (true) {
      auto mut_cluster_instruction = std::make_shared<ClusterInstructionProto>();
      ClusterInstruction::WorkerReceiveInstruction(mut_cluster_instruction.get());
      if (mut_cluster_instruction->has_cluster_ctrl_halt()) {
        break;
      } else if (mut_cluster_instruction->has_cluster_ctrl_abort()) {
        LOG(FATAL) << "received abort instruction";
      } else if (mut_cluster_instruction->has_cluster_ctrl_session_start()) {
        ClusterInstruction::NewSessionBarrier();
        auto wait_session_init = std::make_shared<BlockingCounter>(1);
        wait_session_init_list.emplace_back(wait_session_init);
        AsyncRunLazyJobSet(&lazy_runtime_thread, wait_session_init);
      } else {
        OF_UNIMPLEMENTED();
      }
    }
  }
  ClusterInstruction::HaltBarrier();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
