#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {

Maybe<void> JobBuildAndInferCtxMgr::CreateJobBuildAndInferCtx(const std::string& job_name) {
  if (job_name.empty()) {
    return Maybe<void>(
        GenJobBuildAndInferError(JobBuildAndInferError::kJobNameExists, "job name is empty"));
  }
  if (job_name2infer_ctx_.find(job_name) != job_name2infer_ctx_.end()) {
    return Maybe<void>(GenJobBuildAndInferError(JobBuildAndInferError::kJobNameExists,
                                                "job name: " + job_name + " is exists"));
  }
  job_name2infer_ctx_.emplace(job_name, std::make_shared<JobBuildAndInferCtx>(job_name));
  return Maybe<void>();
}

Maybe<JobBuildAndInferCtx> JobBuildAndInferCtxMgr::FindJobBuildAndInferCtx(
    const std::string& job_name) {
  if (job_name2infer_ctx_.find(job_name) == job_name2infer_ctx_.end()) {
    return Maybe<JobBuildAndInferCtx>(GenJobBuildAndInferError(
        JobBuildAndInferError::kNoJobBuildAndInferCtx, "cannot find job name:" + job_name));
  }
  return Maybe<JobBuildAndInferCtx>(job_name2infer_ctx_.at(job_name));
}

}  // namespace oneflow
