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

Maybe<void> JobBuildAndInferCtx::GenOpProducedEmptyLogicalBlobDesc(Operator* op) {
  std::vector<std::string> produced_bns;
  produced_bns.insert(produced_bns.end(), op->output_bns().begin(), op->output_bns().end());
  produced_bns.insert(produced_bns.end(), op->tmp_bns().begin(), op->tmp_bns().end());
  produced_bns.insert(produced_bns.end(), op->const_buf_bns().begin(), op->const_buf_bns().end());
  for (const std::string& produced_bn : produced_bns) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(produced_bn);
    if (lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end()) {
      return Maybe<void>(GenJobBuildAndInferError(
          JobBuildAndInferError::kLogicalBlobNameRepeated,
          "op_name: " + lbi.op_name() + " blob_name: " + lbi.blob_name() + " repeated"));
    }
    lbi2logical_blob_desc_.emplace(lbi, std::make_unique<BlobDesc>(DataType::kInvalidDataType));
  }
  return Maybe<void>();
}

// TODO(): add handle error of same interface op blob between jobs
Maybe<void> JobBuildAndInferCtx::AddAndInferInputOp(const OperatorConf& op_conf) {
  const std::string& op_name = op_conf.name();
  if (op_name2op_.find(op_name) != op_name2op_.end()) {
    return Maybe<void>(GenJobBuildAndInferError(
        JobBuildAndInferError::kOpNameExists,
        "op_name: " + op_name + "already exists in job: " + job_.job_conf().job_name()));
  }
  if (op_conf.device_type() == DeviceType::kInvalidDevice) {
    return Maybe<void>(GenJobBuildAndInferError(JobBuildAndInferError::kOpConfDeviceTypeNoSet,
                                                "op_name: " + op_name + " not set device type"));
  }
  OperatorConf* mut_op_conf = job_.mutable_net()->add_op();
  *mut_op_conf = op_conf;
  op_name2op_.emplace(op_name, ConstructOp(op_conf));
  Operator* op = op_name2op_.at(op_name).get();
  // TODO() lbn with split hist
  JUST(GenOpProducedEmptyLogicalBlobDesc(op));
  auto GetBlobDesc4BnInOp = [&](const std::string& bn_in_op) -> BlobDesc* {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(bn_in_op);
    if (lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end()) {
      return lbi2logical_blob_desc_.at(lbi).get();
    }
    return nullptr;
  };
  TODO();
}

Maybe<void> JobBuildAndInferCtx::AddAndInferNonInputOp(const OperatorConf& op_conf) {
  if (!has_job_conf_) {
    return Maybe<void>(GenJobBuildAndInferError(JobBuildAndInferError::kJobConfNotSet, ""));
  }
  if (!is_job_conf_frozen_) { is_job_conf_frozen_ = true; }
  return AddAndInferInputOp(op_conf);
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
