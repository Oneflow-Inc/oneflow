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
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/framework/load_library.h"
#include "oneflow/core/job/collective_boxing/scheduler.h"
#include "oneflow/core/job/critical_section_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_set_compile_ctx.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/job/runtime_buffer_managers_scope.h"
#include "oneflow/core/job/runtime_job_descs.h"
#include "oneflow/core/job/version.h"
#include "oneflow/core/memory/chunk_manager.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/user/summary/events_writer.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/graph/task_stream_index_manager.h"

namespace oneflow {

SessionGlobalObjectsScope::SessionGlobalObjectsScope() {}

Maybe<void> SessionGlobalObjectsScope::Init(const ConfigProto& config_proto) {
  session_id_ = config_proto.session_id();
  Global<ResourceDesc, ForSession>::Delete();
  DumpVersionInfo();
  Global<ResourceDesc, ForSession>::New(config_proto.resource(),
                                        GlobalProcessCtx::NumOfProcessPerNode());
  Global<IDMgr>::New();
  Global<TaskStreamIndexManager>::New();
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    Global<JobName2JobId>::New();
    Global<CriticalSectionDesc>::New();
    Global<InterUserJobInfo>::New();
    Global<LazyJobBuildAndInferCtxMgr>::New();
    Global<JobSetCompileCtx>::New();
    Global<RuntimeBufferManagersScope>::New();
  }
  for (const std::string& lib_path : config_proto.load_lib_path()) { JUST(LoadLibrary(lib_path)); }
  {
    // NOTE(chengcheng): Init Global Runtime objects.
    Global<RuntimeCtx>::New();
    Global<MemoryAllocator>::New();
    Global<ChunkMgr>::New();
    Global<RegstMgr>::New();
    Global<ActorMsgBus>::New();
    Global<ThreadMgr>::New();
    Global<RuntimeJobDescs>::New();
    Global<summary::EventsWriter>::New();
    Global<boxing::collective::Scheduler>::New();
  }

  return Maybe<void>::Ok();
}

Maybe<void> SessionGlobalObjectsScope::EagerInit(const ConfigProto& config_proto) {
  session_id_ = config_proto.session_id();
  Global<ResourceDesc, ForSession>::Delete();
  DumpVersionInfo();
  Global<ResourceDesc, ForSession>::New(config_proto.resource());
  for (const std::string& lib_path : config_proto.load_lib_path()) { JUST(LoadLibrary(lib_path)); }
  return Maybe<void>::Ok();
}

SessionGlobalObjectsScope::~SessionGlobalObjectsScope() {
  {
    // NOTE(chengcheng): Delete Global Runtime objects.
    Global<boxing::collective::Scheduler>::Delete();
    Global<summary::EventsWriter>::Delete();
    Global<RuntimeJobDescs>::Delete();
    Global<ThreadMgr>::Delete();
    Global<ActorMsgBus>::Delete();
    Global<RegstMgr>::Delete();
    Global<ChunkMgr>::Delete();
    Global<MemoryAllocator>::Delete();
    Global<RuntimeCtx>::Delete();
  }

  if (GlobalProcessCtx::IsThisProcessMaster()) {
    Global<RuntimeBufferManagersScope>::Delete();
    Global<JobSetCompileCtx>::Delete();
    Global<LazyJobBuildAndInferCtxMgr>::Delete();
    Global<InterUserJobInfo>::Delete();
    Global<CriticalSectionDesc>::Delete();
    Global<JobName2JobId>::Delete();
  }
  Global<TaskStreamIndexManager>::Delete();
  Global<IDMgr>::Delete();
  Global<ResourceDesc, ForSession>::Delete();
  Global<ResourceDesc, ForSession>::New(Global<ResourceDesc, ForEnv>::Get()->resource(),
                                        GlobalProcessCtx::NumOfProcessPerNode());
}

}  // namespace oneflow
