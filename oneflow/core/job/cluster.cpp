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
#include "oneflow/core/control/cluster_control.pb.h"
#include "oneflow/core/control/cluster_control.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/job_set.pb.h"

namespace oneflow {

Maybe<void> Cluster::WorkerLoop() {
  CHECK_OR_RETURN(!Global<MachineCtx>::Get()->IsThisMachineMaster());
  while (ClusterControl::WorkerReceiveHalt() == false) {
    ConfigProto config_proto;
    Global<CtrlClient>::Get()->PullKV("config_proto", &config_proto);
    int32_t machine_num = config_proto.resource().machine_num();
    if (Global<MachineCtx>::Get()->this_machine_id() >= machine_num) { continue; }
    Global<SessionGlobalObjectsScope>::New();
    JUST(Global<SessionGlobalObjectsScope>::Get()->Init(config_proto));

    JobSet job_set;
    Global<CtrlClient>::Get()->PullKV("session_job_set", &job_set);
    {
      Oneflow oneflow;
      JUST(oneflow.Init(job_set));
    }
    Global<SessionGlobalObjectsScope>::Delete();
  }
  ClusterControl::HaltBarrier();
  Global<EnvGlobalObjectsScope>::Delete();
  exit(0);
}

}  // namespace oneflow
