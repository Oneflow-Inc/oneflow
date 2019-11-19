#include <string>

#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"

#if defined(WITH_XLA) || defined(WITH_TENSORRT)
#include "oneflow/xrt/api.h"
#define OF_WITH_XRT
#endif  // WITH_XLA || WITH_TENSORRT

namespace oneflow {

inline void RebuildXrtCompiledJob(const OpGraph& op_graph, Job* job) {
#ifdef OF_WITH_XRT
  const auto& job_desc = GlobalJobDesc();
  TeePersistentLogStream::Create("job_without_xrt_" + std::to_string(job_desc.job_id()))
      ->Write(*job);

  // // Create options to run xrt passes.
  // auto graph = xrt::BuildXrtGraph(&op_graph);
  // auto options = xrt::CreateDefaultXrtPassOptions();
  // options.clustering_options.train_phase = GlobalJobDesc().IsTrain();

  // xrt::RunXrtPass("MarkClusterId", graph.get(), options);
  // xrt::RunXrtPass("BuildSubGraph", graph.get(), options);
  // // Rebuild Job
  // xrt::RunXrtPass("RebuildCompiledJob", graph.get(), options, job);
  xrt::RunCompilationTimeXrtPasses(op_graph, job, job_desc.IsTrain());

  TeePersistentLogStream::Create("job_with_xrt_" + std::to_string(job_desc.job_id()))->Write(*job);
#endif  // OF_WITH_XRT
}

inline bool XrtCompilationEnabled(const JobDesc& job_desc) {
#ifdef OF_WITH_XRT
  if (job_desc.use_xla_jit() != ConfigOption::OPT_UNDEF) {
    xrt::EnableUseXlaJit(job_desc.use_xla_jit());
  }
  if (job_desc.use_tensorrt() != ConfigOption::OPT_UNDEF) {
    xrt::EnableUseTensorRT(job_desc.use_tensorrt());
  }
  return xrt::XrtCompilationEnabled();
#else
  return job_desc.use_xla_jit() == ConfigOption::OPT_ON
         || job_desc.use_tensorrt() == ConfigOption::OPT_ON;
#endif  // OF_WITH_XRT
}

}  // namespace oneflow
