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

#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/framework/load_library.h"
#include "oneflow/core/job/version.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
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

}  // namespace

MultiClientSessionContext::~MultiClientSessionContext() {
  if (is_inited_) {
    {
      // NOTE(chengcheng): delete runtime global objects
      Global<BufferMgr<std::shared_ptr<JobInstance>>>::Delete();
    }

    Global<LazyJobBuildAndInferCtxMgr>::Delete();
    Global<IDMgr>::Delete();

    // TODO(chengcheng): remove ForEnv
    Global<ResourceDesc, ForSession>::Delete();
    Global<ResourceDesc, ForSession>::New(Global<ResourceDesc, ForEnv>::Get()->resource(),
                                          GlobalProcessCtx::NumOfProcessPerNode());
  }
}

Maybe<void> MultiClientSessionContext::LazyInitOnlyOnce(const ConfigProto& config_proto) {
  if (!is_inited_) {
    CHECK_OR_RETURN(GlobalProcessCtx::IsMultiClient());
    DumpVersionInfo();

    Resource resource = config_proto.resource();

    {
      // TODO(chengcheng): remove this hack
      //   env config for multi-client
      resource.set_machine_num(GlobalProcessCtx::NodeSize());
      const int32_t gpu_device_num = GetGpuDeviceNum();
      resource.set_gpu_device_num(gpu_device_num);
      if (gpu_device_num == 0) {
        resource.set_cpu_device_num(GlobalProcessCtx::NumOfProcessPerNode());
      } else {
        resource.set_cpu_device_num(gpu_device_num);
      }
    }

    Global<ResourceDesc, ForSession>::Delete();
    Global<ResourceDesc, ForSession>::New(resource, GlobalProcessCtx::NumOfProcessPerNode());
    Global<IDMgr>::New();
    // TODO(chengcheng): refactor JobBuildAndInferCtxMgr
    Global<LazyJobBuildAndInferCtxMgr>::New();

    for (const std::string& lib_path : config_proto.load_lib_path()) {
      JUST(LoadLibrary(lib_path));
    }

    {
      // NOTE(chengcheng): init runtime global objects
      Global<BufferMgr<std::shared_ptr<JobInstance>>>::New();
    }

    is_inited_ = true;
  }
  return Maybe<void>::Ok();
}

Maybe<int64_t> MultiClientSessionContext::GetJobNameId(const std::string& job_class_name) {
  return job_class_name2id_[job_class_name]++;
}

}  // namespace oneflow
