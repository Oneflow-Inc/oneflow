#ifndef ONEFLOW_PYTHON_JOB_BUILD_AND_INFER_HELPER_H_
#define ONEFLOW_PYTHON_JOB_BUILD_AND_INFER_HELPER_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

struct JobBuildAndInferHelper final {
  static JobBuildAndInferCtx* GetCurInferCtx(std::string* error_str) {
    auto maybe_job_name = TRY(Global<JobBuildAndInferCtxMgr>::Get()->GetCurrentJobName());
    if (maybe_job_name.IsOk() == false) {
      *error_str = PbMessage2TxtString(*maybe_job_name.error());
      return nullptr;
    }
    const std::string& job_name = *maybe_job_name.data();
    auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
    if (maybe_ctx.IsOk() == false) {
      *error_str = PbMessage2TxtString(*maybe_ctx.error());
      return nullptr;
    }
    return maybe_ctx.data().get();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_PYTHON_JOB_BUILD_AND_INFER_HELPER_H_
