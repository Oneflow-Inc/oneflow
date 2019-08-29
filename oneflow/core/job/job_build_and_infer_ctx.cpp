#include "oneflow/core/job/job_build_and_infer_ctx.h"

namespace oneflow {

Error GenJobBuildAndInferError(JobBuildAndInferError err_code, std::string msg) {
  Error err;
  err.set_msg(msg);
  err.set_job_build_and_infer_error(err_code);
  return err;
}

JobBuildAndInferCtx::JobBuildAndInferCtx(const std::string& job_name) {
  is_job_conf_frozen_ = false;
  has_job_conf_ = false;
  job_ = Job();
  job_.mutable_job_conf()->set_job_name(job_name);
}

Maybe<void> JobBuildAndInferCtx::SetJobConf(const JobConfigProto& job_conf) {
  if (is_job_conf_frozen_) {
    return Maybe<void>(GenJobBuildAndInferError(JobBuildAndInferError::kJobConfFrozen, ""));
  }
  if (!has_job_conf_) { has_job_conf_ = true; }
  if (job_.job_conf().job_name() != job_conf.job_name()) {
    return Maybe<void>(GenJobBuildAndInferError(
        JobBuildAndInferError::kJobNameNotEqual,
        "job name you set: " + job_conf.job_name()
            + " not equal to origin job name: " + job_.job_conf().job_name()));
  }
  job_.mutable_job_conf()->CopyFrom(job_conf);
}

Maybe<void> JobBuildAndInferCtx::AddAndInferInputOp(const OperatorConf& op_conf) {
  // TODO(): add check same interface op blob between jobs
  TODO();
}

Maybe<void> JobBuildAndInferCtx::AddAndInferNonInputOp(const OperatorConf& op_conf) {
  if (!has_job_conf_) {
    return Maybe<void>(GenJobBuildAndInferError(JobBuildAndInferError::kJobConfNotSet, ""));
  }
  if (!is_job_conf_frozen_) { is_job_conf_frozen_ = true; }
  TODO();
}

Maybe<void> JobBuildAndInferCtx::AddLossLogicalBlobName(const std::string& lbn) { TODO(); }

bool JobBuildAndInferCtx::HasJobConf() const { return has_job_conf_; }

Maybe<Shape> JobBuildAndInferCtx::GetStaticShape(const std::string& lbn) const { TODO(); }

Maybe<DataType> JobBuildAndInferCtx::GetDataType(const std::string& lbn) const { TODO(); }

Maybe<bool> JobBuildAndInferCtx::GetHasBatchDim(const std::string& lbn) const { TODO(); }

Maybe<bool> JobBuildAndInferCtx::GetHasSplitDim(const std::string& lbn) const { TODO(); }

Maybe<int64_t> JobBuildAndInferCtx::GetSplitDim(const std::string& lbn) const { TODO(); }

Maybe<ParallelDesc> JobBuildAndInferCtx::GetParallelDesc(const std::string& lbn) const { TODO(); }

const Job& JobBuildAndInferCtx::job() const { return job_; }

}  // namespace oneflow
