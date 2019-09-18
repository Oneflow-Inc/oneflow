#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {

Maybe<void> JobBuildAndInferCtxMgr::OpenJobBuildAndInferCtx(const std::string& job_name) {
  CHECK_OR_RETURN(!has_cur_job_) << JobBuildAndInferError::kUnknownJobBuildAndInferError
                                 << "cur job not leave before you enter this job_name:" << job_name;
  CHECK_OR_RETURN(!job_name.empty()) << JobBuildAndInferError::kJobNameEmpty;
  CHECK_OR_RETURN(job_name2infer_ctx_.find(job_name) == job_name2infer_ctx_.end())
      << JobBuildAndInferError::kJobNameExist << "job name: " << job_name << " already exist";
  int64_t job_id = job_set_.job_size();
  Job* job = job_set_.add_job();
  job->mutable_job_conf()->set_job_name(job_name);
  job_name2infer_ctx_.emplace(job_name, std::make_unique<JobBuildAndInferCtx>(job, job_id));
  cur_job_name_ = job_name;
  has_cur_job_ = true;
  return Maybe<void>::Ok();
}

Maybe<JobBuildAndInferCtx*> JobBuildAndInferCtxMgr::FindJobBuildAndInferCtx(
    const std::string& job_name) {
  CHECK_OR_RETURN(job_name2infer_ctx_.find(job_name) != job_name2infer_ctx_.end())
      << JobBuildAndInferError::kNoJobBuildAndInferCtx << "cannot find job name:" << job_name;
  return job_name2infer_ctx_.at(job_name).get();
}

Maybe<std::string> JobBuildAndInferCtxMgr::GetCurrentJobName() {
  CHECK_OR_RETURN(has_cur_job_) << JobBuildAndInferError::kNoJobBuildAndInferCtx
                                << "current has not job name";
  return cur_job_name_;
}

void JobBuildAndInferCtxMgr::CloseCurrentJobBuildAndInferCtx() {
  if (!has_cur_job_) { return; }
  has_cur_job_ = false;
  const JobDesc* job_desc = Global<JobDesc>::Get();
  if (job_desc == nullptr) { return; }
  CHECK_EQ(job_desc->job_name(), cur_job_name_);
  CHECK_EQ(job_desc->job_id(), job_set_.job_size() - 1);
  Global<JobDesc>::Delete();
}

}  // namespace oneflow
