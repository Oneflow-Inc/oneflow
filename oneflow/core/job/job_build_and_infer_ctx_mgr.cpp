#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {

Maybe<void> JobBuildAndInferCtxMgr::EnterJobBuildAndInferContext(const std::string& job_name) {
  if (job_name.empty()) {
    return Maybe<void>(
        GenJobBuildAndInferError(JobBuildAndInferError::kJobNameExists, "job name is empty"));
  }
  if (job_name2infer_ctx_.find(job_name) != job_name2infer_ctx_.end()) {
    return Maybe<void>(GenJobBuildAndInferError(JobBuildAndInferError::kJobNameExists,
                                                "job name: " + job_name + " is exists"));
  }
  job_name2infer_ctx_.emplace(job_name, std::make_shared<JobBuildAndInferCtx>(job_name));
  cur_infer_ctx_ = job_name2infer_ctx_.at(job_name);
  return Maybe<void>();
}

Maybe<JobBuildAndInferCtx> JobBuildAndInferCtxMgr::GetCurrentJobBuildAndInferCtx() {
  if (cur_infer_ctx_.get() == nullptr) {
    return Maybe<JobBuildAndInferCtx>(
        GenJobBuildAndInferError(JobBuildAndInferError::kNoJobBuildAndInferCtx, ""));
  }
  return Maybe<JobBuildAndInferCtx>(cur_infer_ctx_);
}

Maybe<void> JobBuildAndInferCtxMgr::LeaveCurrentJobBuildAndInferCtx() {
  cur_infer_ctx_ = nullptr;
  return Maybe<void>();
}

}  // namespace oneflow
