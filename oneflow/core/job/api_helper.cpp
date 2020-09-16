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
#include "oneflow/core/job/api_helper.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/session_global_objects_scope.h"

namespace oneflow {

Maybe<void> StartGlobalSession() {
  CHECK_NOTNULL_OR_RETURN(Global<SessionGlobalObjectsScope>::Get()) << "session not found";
  CHECK_OR_RETURN(Global<MachineCtx>::Get()->IsThisMachineMaster());
  const JobSet& job_set = Global<LazyJobBuildAndInferCtxMgr>::Get()->job_set();
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    TeePersistentLogStream::Create("job_set.prototxt")->Write(job_set);
  }
  if (job_set.job().empty()) { return Error::JobSetEmptyError() << "no function defined"; }
  CHECK_ISNULL_OR_RETURN(Global<Oneflow>::Get());
  Global<CtrlClient>::Get()->PushKV("session_job_set", job_set);
  Global<const InterJobReuseMemStrategy>::New(job_set.inter_job_reuse_mem_strategy());
  Global<Oneflow>::New();
  JUST(Global<Oneflow>::Get()->Init(job_set));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
