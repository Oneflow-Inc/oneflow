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
  Singleton<ResourceDesc, ForSession>::Delete();
  DumpVersionInfo();
  Singleton<ResourceDesc, ForSession>::New(config_proto.resource(),
                                           GlobalProcessCtx::NumOfProcessPerNode());
  Singleton<IDMgr>::New();
  Singleton<TaskStreamIndexManager>::New();
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    Singleton<JobName2JobId>::New();
    Singleton<CriticalSectionDesc>::New();
    Singleton<InterUserJobInfo>::New();
    Singleton<LazyJobBuildAndInferCtxMgr>::New();
    Singleton<JobSetCompileCtx>::New();
    Singleton<RuntimeBufferManagersScope>::New();
  }
  for (const std::string& lib_path : config_proto.load_lib_path()) { JUST(LoadLibrary(lib_path)); }
  {
    // NOTE(chengcheng): Init Global(singleton) Runtime objects.
    Singleton<RuntimeCtx>::New();
    Singleton<MemoryAllocator>::New();
    Singleton<ChunkMgr>::New();
    Singleton<RegstMgr>::New();
    Singleton<ActorMsgBus>::New();
    Singleton<ThreadMgr>::New();
    Singleton<RuntimeJobDescs>::New();
    Singleton<summary::EventsWriter>::New();
    Singleton<boxing::collective::Scheduler>::New();
  }

  return Maybe<void>::Ok();
}

Maybe<void> SessionGlobalObjectsScope::EagerInit(const ConfigProto& config_proto) {
  session_id_ = config_proto.session_id();
  Singleton<ResourceDesc, ForSession>::Delete();
  DumpVersionInfo();
  Singleton<ResourceDesc, ForSession>::New(config_proto.resource());
  for (const std::string& lib_path : config_proto.load_lib_path()) { JUST(LoadLibrary(lib_path)); }
  return Maybe<void>::Ok();
}

SessionGlobalObjectsScope::~SessionGlobalObjectsScope() {
  {
    // NOTE(chengcheng): Delete Global(singleton) Runtime objects.
    Singleton<boxing::collective::Scheduler>::Delete();
    Singleton<summary::EventsWriter>::Delete();
    Singleton<RuntimeJobDescs>::Delete();
    Singleton<ThreadMgr>::Delete();
    Singleton<ActorMsgBus>::Delete();
    Singleton<RegstMgr>::Delete();
    Singleton<ChunkMgr>::Delete();
    Singleton<MemoryAllocator>::Delete();
    Singleton<RuntimeCtx>::Delete();
  }

  if (GlobalProcessCtx::IsThisProcessMaster()) {
    Singleton<RuntimeBufferManagersScope>::Delete();
    Singleton<JobSetCompileCtx>::Delete();
    Singleton<LazyJobBuildAndInferCtxMgr>::Delete();
    Singleton<InterUserJobInfo>::Delete();
    Singleton<CriticalSectionDesc>::Delete();
    Singleton<JobName2JobId>::Delete();
  }
  Singleton<TaskStreamIndexManager>::Delete();
  Singleton<IDMgr>::Delete();
  Singleton<ResourceDesc, ForSession>::Delete();
  Singleton<ResourceDesc, ForSession>::New(Singleton<ResourceDesc, ForEnv>::Get()->resource(),
                                           GlobalProcessCtx::NumOfProcessPerNode());
}

}  // namespace oneflow
