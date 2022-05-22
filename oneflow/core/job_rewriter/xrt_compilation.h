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
#include <string>

#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/global_for.h"

#if defined(WITH_XLA) || defined(WITH_TENSORRT) || defined(WITH_OPENVINO)
#include "oneflow/xrt/api.h"
#define OF_WITH_XRT
#endif  // WITH_XLA || WITH_TENSORRT

namespace oneflow {

inline Maybe<void> RebuildXrtCompiledJob(const OpGraph& op_graph, Job* job) {
#ifdef OF_WITH_XRT
  const auto& job_desc = GlobalJobDesc();
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    TeePersistentLogStream::Create("job_without_xrt_" + std::to_string(job_desc.job_id()))
        ->Write(*job);
  }
  // Run compilation time passes currently include `MarkClusterId`, `BuildSubGraph`
  // and `RebuildCompiledJob`.
  xrt::RunCompilationTimeXrtPasses(op_graph, job, job_desc.IsTrain());

  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    TeePersistentLogStream::Create("job_with_xrt_" + std::to_string(job_desc.job_id()))
        ->Write(*job);
  }
#endif  // OF_WITH_XRT
  return Maybe<void>::Ok();
}

inline bool XrtCompilationEnabled(const JobDesc& job_desc) {
  bool xrt_compilation_enabled = false;

#ifdef OF_WITH_XRT
  xrt_compilation_enabled = xrt::XrtCompilationEnabled();
#endif

  if (job_desc.has_xrt_config()) {
    const XrtConfig& config = job_desc.xrt_config();
    return xrt_compilation_enabled || (config.has_use_xla_jit() && config.use_xla_jit())
           || (config.has_use_tensorrt() && config.use_tensorrt())
           || (config.has_use_openvino() && config.use_openvino());
  }
  return xrt_compilation_enabled;
}

}  // namespace oneflow
