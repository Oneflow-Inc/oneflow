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
#include "api.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/job/cluster_instruction.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/control/ctrl_bootstrap.h"
#include "oneflow/core/rpc/include/base.h"
#include "oneflow/core/vm/vm_util.h"

namespace ofapi {

namespace of = oneflow;

namespace {  // for inltialize

inline bool isEnvInited() { return of::Global<of::EnvGlobalObjectsScope>::Get() != nullptr; }

of::Maybe<void> initEnv() {
  const auto completeEnvProto = [](of::EnvProto& env_proto) {
    auto bootstrap_conf = env_proto.mutable_ctrl_bootstrap_conf();
    auto master_addr = bootstrap_conf->mutable_master_addr();
    master_addr->set_host("127.0.0.1");
    master_addr->set_port(38799);
    bootstrap_conf->set_world_size(1);
    bootstrap_conf->set_rank(0);
  };

  of::EnvProto env_proto;
  completeEnvProto(env_proto);
  of::Global<of::EnvGlobalObjectsScope>::SetAllocated(new of::EnvGlobalObjectsScope());
  JUST(of::Global<of::EnvGlobalObjectsScope>::Get()->Init(env_proto));
  return of::Maybe<void>::Ok();
}

}  // namespace

void initialize() {
  of::SetIsMultiClient(true).GetOrThrow();
  if (!isEnvInited()) { initEnv().GetOrThrow(); }
}

void release() {
  if (isEnvInited()) {
    // sync multi_client
    of::vm::ClusterSync().GetOrThrow();
    // destory env
    if (of::IsMultiClient().GetOrThrow()) {
      OF_ENV_BARRIER();
    } else {
      of::ClusterInstruction::MasterSendHalt();
    }
    of::Global<of::EnvGlobalObjectsScope>::Delete();
  }
  // TODO close session
  of::SetShuttingDown();
}

}  // namespace ofapi
