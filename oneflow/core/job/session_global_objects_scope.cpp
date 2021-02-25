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
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/available_memory_desc.pb.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/profiler.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/foreign_job_instance.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/critical_section_desc.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_set_compile_ctx.h"
#include "oneflow/core/job/runtime_buffer_managers_scope.h"
#include "oneflow/core/framework/load_library.h"
#include "oneflow/core/job/version.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

namespace {

std::string GetAmdCtrlKey(int64_t machine_id) {
  return "AvailableMemDesc/" + std::to_string(machine_id);
}

void PushAvailableMemDescOfThisMachine() {
  AvailableMemDescOfMachine this_machine_mem_desc;
#ifdef WITH_CUDA
  FOR_RANGE(int, i, 0, (Global<ResourceDesc, ForSession>::Get()->GpuDeviceNum())) {
    this_machine_mem_desc.add_zone_size(GetAvailableGpuMemSize(i));
  }
#endif
  this_machine_mem_desc.add_zone_size(GetAvailableCpuMemSize());
  Global<CtrlClient>::Get()->PushKV(GetAmdCtrlKey(GlobalProcessCtx::Rank()), this_machine_mem_desc);
}

AvailableMemDesc PullAvailableMemDesc() {
  AvailableMemDesc ret;
  AvailableMemDescOfMachine machine_amd_i;
  FOR_RANGE(int64_t, i, 0, (Global<ResourceDesc, ForSession>::Get()->TotalMachineNum())) {
    Global<CtrlClient>::Get()->PullKV(GetAmdCtrlKey(i), ret.add_machine_amd());
  }
  return ret;
}

}  // namespace

SessionGlobalObjectsScope::SessionGlobalObjectsScope() {}

Maybe<void> SessionGlobalObjectsScope::Init(const ConfigProto& config_proto) {
  session_id_ = config_proto.session_id();
  Global<ResourceDesc, ForSession>::Delete();
  DumpVersionInfo();
  Global<ResourceDesc, ForSession>::New(config_proto.resource());
  Global<const IOConf>::New(config_proto.io_conf());
  Global<const IOConf>::SessionNew(config_proto.session_id(), config_proto.io_conf());
  Global<const ProfilerConf>::New(config_proto.profiler_conf());
  Global<IDMgr>::New();
  if (GlobalProcessCtx::IsThisProcessMaster()
      && Global<const ProfilerConf>::Get()->collect_act_event()) {
    Global<Profiler>::New();
  }
  PushAvailableMemDescOfThisMachine();
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    Global<AvailableMemDesc>::New();
    *Global<AvailableMemDesc>::Get() = PullAvailableMemDesc();
    Global<JobName2JobId>::New();
    Global<CriticalSectionDesc>::New();
    Global<InterUserJobInfo>::New();
    Global<LazyJobBuildAndInferCtxMgr>::New();
    Global<JobSetCompileCtx>::New();
    Global<RuntimeBufferManagersScope>::New();
  }
  for (const std::string lib_path : config_proto.load_lib_path()) { JUST(LoadLibrary(lib_path)); }
  return Maybe<void>::Ok();
}

SessionGlobalObjectsScope::~SessionGlobalObjectsScope() {
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    Global<RuntimeBufferManagersScope>::Delete();
    Global<JobSetCompileCtx>::Delete();
    Global<LazyJobBuildAndInferCtxMgr>::Delete();
    Global<InterUserJobInfo>::Delete();
    Global<CriticalSectionDesc>::Delete();
    Global<JobName2JobId>::Delete();
    Global<AvailableMemDesc>::Delete();
  }
  if (Global<Profiler>::Get() != nullptr) { Global<Profiler>::Delete(); }
  Global<IDMgr>::Delete();
  Global<const ProfilerConf>::Delete();
  Global<const IOConf>::Delete();
  Global<const IOConf>::SessionDelete(session_id_);
  Global<ResourceDesc, ForSession>::Delete();
  Global<ResourceDesc, ForSession>::New(Global<ResourceDesc, ForEnv>::Get()->resource());
}

}  // namespace oneflow
