#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {

Maybe<void> JobBuildAndInferCtxMgr::EnterJobBuildAndInferCtx(const std::string& job_name) {
  if (has_cur_job_) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kUnknownJobBuildAndInferError,
                                    "cur job not leave before you enter this job_name:" + job_name);
  }
  if (job_name.empty()) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kJobNameEmpty, "job name is empty");
  }
  if (job_name2infer_ctx_.find(job_name) != job_name2infer_ctx_.end()) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kJobNameExist,
                                    "job name: " + job_name + " already exist");
  }
  job_name2infer_ctx_.emplace(job_name, std::make_shared<JobBuildAndInferCtx>(job_name));
  cur_job_name_ = job_name;
  has_cur_job_ = true;
  return Maybe<void>();
}

Maybe<JobBuildAndInferCtx> JobBuildAndInferCtxMgr::FindJobBuildAndInferCtx(
    const std::string& job_name) {
  if (job_name2infer_ctx_.find(job_name) == job_name2infer_ctx_.end()) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kNoJobBuildAndInferCtx,
                                    "cannot find job name:" + job_name);
  }
  return job_name2infer_ctx_.at(job_name);
}

Maybe<std::string> JobBuildAndInferCtxMgr::GetCurrentJobName() {
  if (!has_cur_job_) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kNoJobBuildAndInferCtx,
                                    "current has not job name");
  }
  return cur_job_name_;
}

void JobBuildAndInferCtxMgr::LeaveCurrentJobBuildAndInferCtx() { has_cur_job_ = false; }

}  // namespace oneflow
