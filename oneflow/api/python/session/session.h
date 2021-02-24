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
#ifndef ONEFLOW_API_PYTHON_SESSION_SESSION_H_
#define ONEFLOW_API_PYTHON_SESSION_SESSION_H_

#include <string>
#include <google/protobuf/text_format.h>
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/cluster_instruction.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

inline Maybe<bool> IsSessionInited() { return Global<SessionGlobalObjectsScope>::Get() != nullptr; }

inline void FixCpuDeviceNum(ConfigProto* config_proto) {
  if (config_proto->resource().cpu_device_num() > 0) { return; }
  config_proto->mutable_resource()->set_cpu_device_num(std::thread::hardware_concurrency());
}

inline Maybe<void> InitLazyGlobalSession(const std::string& config_proto_str) {
  CHECK_NOTNULL_OR_RETURN(Global<EnvDesc>::Get()) << "env not found";
  CHECK_OR_RETURN(GlobalProcessCtx::IsThisProcessMaster());

  ClusterInstruction::MasterSendSessionStart();

  ConfigProto config_proto;
  CHECK_OR_RETURN(TxtString2PbMessage(config_proto_str, &config_proto))
      << "failed to parse config_proto: " << config_proto_str;
  FixCpuDeviceNum(&config_proto);
  Global<CtrlClient>::Get()->PushKV("config_proto", config_proto);

  CHECK_ISNULL_OR_RETURN(Global<SessionGlobalObjectsScope>::Get());
  Global<SessionGlobalObjectsScope>::SetAllocated(new SessionGlobalObjectsScope());
  JUST(Global<SessionGlobalObjectsScope>::Get()->Init(config_proto));
  LOG(INFO) << "NewGlobal " << typeid(SessionGlobalObjectsScope).name();
  return Maybe<void>::Ok();
}

inline Maybe<void> DestroyLazyGlobalSession() {
  if (Global<SessionGlobalObjectsScope>::Get() == nullptr) { return Maybe<void>::Ok(); }
  CHECK_OR_RETURN(GlobalProcessCtx::IsThisProcessMaster());
  Global<SessionGlobalObjectsScope>::Delete();
  return Maybe<void>::Ok();
}

inline Maybe<void> StartLazyGlobalSession() {
  CHECK_NOTNULL_OR_RETURN(Global<SessionGlobalObjectsScope>::Get()) << "session not found";
  CHECK_OR_RETURN(GlobalProcessCtx::IsThisProcessMaster());
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

inline Maybe<void> StopLazyGlobalSession() {
  if (Global<Oneflow>::Get() == nullptr) { return Maybe<void>::Ok(); }
  CHECK_OR_RETURN(GlobalProcessCtx::IsThisProcessMaster());
  CHECK_NOTNULL_OR_RETURN(Global<Oneflow>::Get());
  Global<Oneflow>::Delete();
  Global<const InterJobReuseMemStrategy>::Delete();
  return Maybe<void>::Ok();
}

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_SESSION_SESSION_H_
