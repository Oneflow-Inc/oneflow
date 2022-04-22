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

#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/framework/load_library.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/version.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/critical_section_instance.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/job/runtime_job_descs.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/user/summary/events_writer.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/memory/chunk_manager.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/job/collective_boxing/scheduler.h"
#include "oneflow/core/graph/task_stream_index_manager.h"
#include "oneflow/core/framework/variable_tensor_mgr.h"
#ifdef WITH_CUDA
#include <cuda.h>
#endif  // WITH_CUDA

namespace oneflow {

namespace {

int32_t GetGpuDeviceNum() {
#ifndef WITH_CUDA
  return 0;
#else
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  return device_count;
#endif
}

int32_t GetCpuDeviceNum() { return std::thread::hardware_concurrency(); }

}  // namespace

MultiClientSessionContext::MultiClientSessionContext(
    const std::shared_ptr<EnvGlobalObjectsScope>& env_ctx)
    : env_ctx_(env_ctx) {
  CHECK(Global<MultiClientSessionContext>::Get() == nullptr);
  Global<MultiClientSessionContext>::SetAllocated(this);
}

MultiClientSessionContext::~MultiClientSessionContext() {
  CHECK_JUST(TryClose());
  if (Global<MultiClientSessionContext>::Get() != nullptr) {
    Global<MultiClientSessionContext>::SetAllocated(nullptr);
  }
}

Maybe<void> MultiClientSessionContext::TryInit(const ConfigProto& config_proto) {
  if (!is_inited_) {
    DumpVersionInfo();

    Resource resource = config_proto.resource();

    {
      // NOTE(chengcheng):
      //   In multi-client, user can NOT config gpu_device_num and cpu_device_num.
      //
      //   cpu_device_num is a confusing name, it should be explained as:
      //       in current rank, assign CPU actor compute stream in this optional range.
      //       That is, the number of independent CPU devices that can be abstracted from
      //       this machine and this process.
      //   gpu_device_num is the number of visible GPUs one current machine.
      //
      //   NOTE: gpu_device_num and cpu_device_num NOT necessarily equal to the num of process
      //       on this machine.
      resource.set_machine_num(GlobalProcessCtx::NodeSize());
      resource.set_gpu_device_num(GetGpuDeviceNum());
      resource.set_cpu_device_num(GetCpuDeviceNum());
    }

    // NOTE(chengcheng): detele first because in EnvGlobalObjectScope has created ResourceDesc.
    if (Global<ResourceDesc, ForSession>::Get() != nullptr) {
      // TODO(chengcheng): reorganize dependency of all Global objects.
      Global<ResourceDesc, ForSession>::Delete();
    }
    Global<ResourceDesc, ForSession>::New(resource, GlobalProcessCtx::NumOfProcessPerNode());
    Global<IDMgr>::New();
    Global<TaskStreamIndexManager>::New();
    // TODO(chengcheng): refactor JobBuildAndInferCtxMgr
    Global<LazyJobBuildAndInferCtxMgr>::New();

    for (const std::string& lib_path : config_proto.load_lib_path()) {
      // TODO(chengcheng): remove load_lib_path in config proto. using LoadLibraryNow
      JUST(LoadLibrary(lib_path));
    }

    {
      // NOTE(chengcheng): init runtime global objects
      Global<BufferMgr<std::shared_ptr<JobInstance>>>::New();
      Global<BufferMgr<std::shared_ptr<CriticalSectionInstance>>>::New();
      Global<RuntimeCtx>::New();
      Global<MemoryAllocator>::New();
      Global<ChunkMgr>::New();
      Global<RegstMgr>::New();
      Global<ActorMsgBus>::New();
      Global<ThreadMgr>::New();
      Global<RuntimeJobDescs>::New();
      Global<summary::EventsWriter>::New();
      Global<boxing::collective::Scheduler>::New();
      Global<VariableTensorMgr>::New();
    }

    is_inited_ = true;
  }
  return Maybe<void>::Ok();
}

Maybe<void> MultiClientSessionContext::TryInit(const std::string& config_proto_str) {
  ConfigProto config_proto;
  CHECK_OR_RETURN(TxtString2PbMessage(config_proto_str, &config_proto))
      << "failed to parse config_proto: " << config_proto_str;
  return TryInit(config_proto);
}

Maybe<void> MultiClientSessionContext::UpdateResource(const Resource& reso_proto) {
  CHECK_OR_RETURN(is_inited_) << " session must be inited when updating resource.";
  CHECK_NOTNULL_OR_RETURN((Global<ResourceDesc, ForSession>::Get()));
  Global<ResourceDesc, ForSession>::Get()->Update(reso_proto);
  return Maybe<void>::Ok();
}

Maybe<void> MultiClientSessionContext::UpdateResource(const std::string& reso_proto_str) {
  Resource reso_proto;
  CHECK_OR_RETURN(TxtString2PbMessage(reso_proto_str, &reso_proto))
      << "failed to parse config_proto: " << reso_proto_str;
  return UpdateResource(reso_proto);
}

Maybe<void> MultiClientSessionContext::TryClose() {
  if (is_inited_) {
    VLOG(1) << "Try to delete multi client session context." << std::endl;
    {
      // NOTE(chengcheng): delete runtime global objects
      Global<boxing::collective::Scheduler>::Delete();
      Global<summary::EventsWriter>::Delete();
      Global<RuntimeJobDescs>::Delete();
      Global<ThreadMgr>::Delete();
      Global<ActorMsgBus>::Delete();
      Global<RegstMgr>::Delete();
      Global<ChunkMgr>::Delete();
      Global<MemoryAllocator>::Delete();
      Global<RuntimeCtx>::Delete();
      Global<BufferMgr<std::shared_ptr<CriticalSectionInstance>>>::Delete();
      Global<BufferMgr<std::shared_ptr<JobInstance>>>::Delete();
      Global<VariableTensorMgr>::Delete();
    }

    Global<LazyJobBuildAndInferCtxMgr>::Delete();
    Global<TaskStreamIndexManager>::Delete();
    Global<IDMgr>::Delete();

    // TODO(chengcheng): remove template ForEnv and ForSession
    Global<ResourceDesc, ForSession>::Delete();
    // NOTE(chengcheng): New after delete because in EnvGlobalObjectScope once created ResourceDesc.
    Global<ResourceDesc, ForSession>::New(Global<ResourceDesc, ForEnv>::Get()->resource(),
                                          GlobalProcessCtx::NumOfProcessPerNode());
    VLOG(1) << "Finish delete multi client session context." << std::endl;
    env_ctx_.reset();
    is_inited_ = false;
  }
  return Maybe<void>::Ok();
}

void MultiClientSessionContext::StoreFreeEagerTensorWithNameByGraphName(
    const std::string& graph_name, const std::shared_ptr<one::Tensor>& tensor,
    const std::string& tensor_name) {
  auto it = graph_name2free_eager_tensors_.find(graph_name);
  if (it == graph_name2free_eager_tensors_.end()) {
    it = graph_name2free_eager_tensors_
             .emplace(graph_name,
                      std::vector<std::pair<std::string, std::shared_ptr<one::Tensor>>>())
             .first;
  }
  it->second.emplace_back(std::make_pair(tensor_name, tensor));
}

const std::vector<std::pair<std::string, std::shared_ptr<one::Tensor>>>&
MultiClientSessionContext::GetFreeEagerTensorNamePairByGraphName(const std::string& graph_name) {
  auto it = graph_name2free_eager_tensors_.find(graph_name);
  if (it == graph_name2free_eager_tensors_.end()) {
    it = graph_name2free_eager_tensors_
             .emplace(graph_name,
                      std::vector<std::pair<std::string, std::shared_ptr<one::Tensor>>>())
             .first;
  }
  return it->second;
}

void MultiClientSessionContext::RemoveGraphFreeEagerTensors(const std::string& graph_name) {
  graph_name2free_eager_tensors_.erase(graph_name);
}

}  // namespace oneflow
