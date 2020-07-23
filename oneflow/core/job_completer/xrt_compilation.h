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

#if defined(WITH_XLA) || defined(WITH_TENSORRT)
#include "oneflow/xrt/api.h"
#define OF_WITH_XRT
#endif  // WITH_XLA || WITH_TENSORRT

namespace oneflow {

inline void RebuildXrtCompiledJob(const OpGraph& op_graph, Job* job) {
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
}

inline bool XrtCompilationEnabled(const JobDesc& job_desc) {
  if (!job_desc.has_xrt_config()) {
#ifdef OF_WITH_XRT
    return xrt::XrtCompilationEnabled();
#else
    return false;
#endif  // OF_WITH_XRT
  }
  const XrtConfig& config = job_desc.xrt_config();
#ifdef OF_WITH_XRT
  xrt::InitXrtConfigurations(config);
  return xrt::XrtCompilationEnabled();
#else
  return (config.has_use_xla_jit() && config.use_xla_jit())
         || (config.has_use_tensorrt() && config.use_tensorrt());
#endif  // OF_WITH_XRT
}

}  // namespace oneflow
