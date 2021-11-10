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
#include "oneflow/core/device/node_device_descriptor_manager.h"
#include "oneflow/core/framework/load_library.h"
#include "oneflow/core/job/available_memory_desc.pb.h"
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

#ifdef WITH_CUDA
#include "oneflow/core/device/cuda_device_descriptor.h"
#endif  // WITH_CUDA

namespace oneflow {

namespace {

AvailableMemDescOfMachine GetAvailableMemDescOfMachine(int64_t rank) {
  AvailableMemDescOfMachine machine_mem_desc;
  const auto node_desc =
      Global<device::NodeDeviceDescriptorManager>::Get()->GetNodeDeviceDescriptor(rank);
#ifdef WITH_CUDA
  const auto cuda_device_list =
      node_desc->GetDeviceDescriptorList(device::kCudaDeviceDescriptorClassName);
  CHECK(cuda_device_list);
  FOR_RANGE(int, i, 0, (Global<ResourceDesc, ForSession>::Get()->GpuDeviceNum())) {
    if (i >= cuda_device_list->DeviceCount()) {
      LOG(WARNING) << "Invalid CUDA device ordinal: rank " << rank << " ordinal " << i;
      machine_mem_desc.add_zone_size(0);
    } else {
      const auto cuda_device = std::dynamic_pointer_cast<const device::CudaDeviceDescriptor>(
          cuda_device_list->GetDevice(i));
      CHECK(cuda_device);
      machine_mem_desc.add_zone_size(cuda_device->GlobalMemorySizeBytes());
    }
  }
#endif
  machine_mem_desc.add_zone_size(node_desc->HostMemorySizeBytes());
  return machine_mem_desc;
}

AvailableMemDesc GetAvailableMemDesc() {
  AvailableMemDesc ret;
  for (int64_t i : Global<ResourceDesc, ForSession>::Get()->process_ranks()) {
    *ret.add_machine_amd() = GetAvailableMemDescOfMachine(i);
  }
  return ret;
}

AvailableMemDesc GetDryRunAvailableMemDesc() {
  AvailableMemDescOfMachine this_machine_mem_desc;
#ifdef WITH_CUDA
  FOR_RANGE(int, i, 0, (Global<ResourceDesc, ForSession>::Get()->GpuDeviceNum())) {
    this_machine_mem_desc.add_zone_size(std::numeric_limits<size_t>::max());
  }
#endif
  this_machine_mem_desc.add_zone_size(std::numeric_limits<size_t>::max());

  AvailableMemDesc ret;
  AvailableMemDescOfMachine machine_amd_i;
  for (int64_t i = 0; i < Global<ResourceDesc, ForSession>::Get()->process_ranks().size(); ++i) {
    *ret.add_machine_amd() = this_machine_mem_desc;
  }
  return ret;
}

}  // namespace

SessionGlobalObjectsScope::SessionGlobalObjectsScope() {}

Maybe<void> SessionGlobalObjectsScope::Init(const ConfigProto& config_proto) {
  session_id_ = config_proto.session_id();
  Global<ResourceDesc, ForSession>::Delete();
  DumpVersionInfo();
  Global<ResourceDesc, ForSession>::New(config_proto.resource(),
                                        GlobalProcessCtx::NumOfProcessPerNode());
  Global<IDMgr>::New();
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    Global<AvailableMemDesc>::New();
    if (Global<ResourceDesc, ForSession>::Get()->enable_dry_run()) {
      *Global<AvailableMemDesc>::Get() = GetDryRunAvailableMemDesc();
    } else {
      *Global<AvailableMemDesc>::Get() = GetAvailableMemDesc();
    }
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
    Global<AvailableMemDesc>::Delete();
  }
  Global<IDMgr>::Delete();
  Global<ResourceDesc, ForSession>::Delete();
  Global<ResourceDesc, ForSession>::New(Global<ResourceDesc, ForEnv>::Get()->resource(),
                                        GlobalProcessCtx::NumOfProcessPerNode());
}

}  // namespace oneflow
