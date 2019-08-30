#include "oneflow/core/job/job_build_and_infer_ctx.h"

namespace oneflow {

Error GenJobBuildAndInferError(JobBuildAndInferError err_code, std::string msg) {
  Error err;
  err.set_msg(msg);
  err.set_job_build_and_infer_error(err_code);
  return err;
}

JobBuildAndInferCtx::JobBuildAndInferCtx(Job* job) : job_(job) {
  is_job_conf_frozen_ = false;
  has_job_conf_ = false;
}

Maybe<void> JobBuildAndInferCtx::SetJobConf(const JobConfigProto& job_conf) {
  if (is_job_conf_frozen_) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kJobConfFrozen, "");
  }
  if (has_job_conf_) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kJobConfRepeatedSet, "");
  }
  has_job_conf_ = true;
  if (job_->job_conf().job_name() != job_conf.job_name()) {
    return GenJobBuildAndInferError(
        JobBuildAndInferError::kJobNameNotEqual,
        "job name you set: " + job_conf.job_name()
            + " not equal to origin job name: " + job_->job_conf().job_name());
  }
  job_->mutable_job_conf()->CopyFrom(job_conf);
  return Maybe<void>();
}

Maybe<void> JobBuildAndInferCtx::GenOpProducedEmptyLogicalBlobDesc(Operator* op) {
  // check consumed blob
  for (const std::string& consumed_bn : op->input_bns()) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(consumed_bn);
    if (lbi2logical_blob_desc_.find(lbi) == lbi2logical_blob_desc_.end()) {
      return GenJobBuildAndInferError(JobBuildAndInferError::kLogicalBlobNameNotExist,
                                      "op_name: " + op->op_name()
                                          + " consumed_op_name:" + lbi.op_name()
                                          + " blob_name: " + lbi.blob_name() + " not exist");
    }
  }

  // create produced blob
  std::vector<std::string> produced_bns;
  produced_bns.insert(produced_bns.end(), op->output_bns().begin(), op->output_bns().end());
  produced_bns.insert(produced_bns.end(), op->tmp_bns().begin(), op->tmp_bns().end());
  produced_bns.insert(produced_bns.end(), op->const_buf_bns().begin(), op->const_buf_bns().end());
  for (const std::string& produced_bn : produced_bns) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(produced_bn);
    if (lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end()) {
      return GenJobBuildAndInferError(
          JobBuildAndInferError::kLogicalBlobNameRepeated,
          "op_name: " + lbi.op_name() + " blob_name: " + lbi.blob_name() + " is repeated");
    }
    lbi2logical_blob_desc_.emplace(lbi, std::make_unique<BlobDesc>(DataType::kInvalidDataType));
  }
  return Maybe<void>();
}

// TODO(): add handle error of same interface op blob between jobs
Maybe<void> JobBuildAndInferCtx::AddAndInferInputOp(const OperatorConf& op_conf) {
  const std::string& op_name = op_conf.name();
  if (op_name2op_.find(op_name) != op_name2op_.end()) {
    return GenJobBuildAndInferError(
        JobBuildAndInferError::kOpNameExist,
        "op_name: " + op_name + "already exist in job: " + job_->job_conf().job_name());
  }
  if (op_conf.device_type() == DeviceType::kInvalidDevice) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kOpConfDeviceTypeNoSet,
                                    "op_name: " + op_name + " not set device type");
  }
  OperatorConf* mut_op_conf = job_->mutable_net()->add_op();
  *mut_op_conf = op_conf;
  op_name2op_.emplace(op_name, ConstructOp(op_conf));
  Operator* op = op_name2op_.at(op_name).get();
  // TODO() lbn with split hist
  JUST(GenOpProducedEmptyLogicalBlobDesc(op));
  auto GetBlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(bn);
    if (lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end()) {
      return lbi2logical_blob_desc_.at(lbi).get();
    }
    return nullptr;
  };
  ParallelContext parallel_ctx;
  parallel_ctx.set_parallel_id(0);
  parallel_ctx.set_parallel_num(1);
  parallel_ctx.set_policy(ParallelPolicy::kDataParallel);
  op->InferOutBlobDescsIf(GetBlobDesc4BnInOp, &parallel_ctx, job_->job_conf().piece_size(),
                          [](OpContext*) {});
  auto HasBatchDim4BnInOp = [&](const std::string& bn) -> bool* {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(bn);
    return &(lbi2has_batch_dim_[lbi]);
  };
  auto GetConstBlobDescBnInOp = [&](const std::string& bn) -> const BlobDesc& {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(bn);
    return *(lbi2logical_blob_desc_[lbi].get());
  };
  op->InferHasBatchDimIf(GetConstBlobDescBnInOp, HasBatchDim4BnInOp);
  // TODO()  infer blob desc split dim
  return Maybe<void>();
}

Maybe<void> JobBuildAndInferCtx::AddAndInferNonInputOp(const OperatorConf& op_conf) {
  if (!has_job_conf_) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kJobConfNotSet, "");
  }
  if (!is_job_conf_frozen_) { is_job_conf_frozen_ = true; }
  return AddAndInferInputOp(op_conf);
}

Maybe<void> JobBuildAndInferCtx::AddLossLogicalBlobName(const std::string& lbn) {
  if (!(job_->job_conf().has_train_conf())) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kUnknownJobBuildAndInferError,
                                    "job has not TrainConf when add loss logical blob name");
  }
  job_->mutable_job_conf()->mutable_train_conf()->add_loss_lbn(lbn);
  return Maybe<void>();
}

bool JobBuildAndInferCtx::HasJobConf() const { return has_job_conf_; }

#define GEN_ERROR_WHEN_GET_INFO_FROM_LBN(info_src)                                                 \
  if (lbn.find('/') == std::string::npos) {                                                        \
    return GenJobBuildAndInferError(JobBuildAndInferError::kLogicalBlobNameInvalid, "lbn:" + lbn); \
  }                                                                                                \
  LogicalBlobId lbi = GenLogicalBlobId(lbn);                                                       \
  if (info_src.find(lbi) == info_src.end()) {                                                      \
    return GenJobBuildAndInferError(JobBuildAndInferError::kLogicalBlobNameNotExist,               \
                                    "lbn:" + lbn);                                                 \
  }

Maybe<Shape> JobBuildAndInferCtx::GetStaticShape(const std::string& lbn) const {
  GEN_ERROR_WHEN_GET_INFO_FROM_LBN(lbi2logical_blob_desc_);
  return lbi2logical_blob_desc_.at(lbi)->shape();
}

Maybe<DataType> JobBuildAndInferCtx::GetDataType(const std::string& lbn) const {
  GEN_ERROR_WHEN_GET_INFO_FROM_LBN(lbi2logical_blob_desc_);
  return lbi2logical_blob_desc_.at(lbi)->data_type();
}

Maybe<bool> JobBuildAndInferCtx::GetHasBatchDim(const std::string& lbn) const {
  GEN_ERROR_WHEN_GET_INFO_FROM_LBN(lbi2has_batch_dim_);
  return lbi2has_batch_dim_.at(lbi);
}

Maybe<bool> JobBuildAndInferCtx::GetHasSplitDimFromProducerView(const std::string& lbn) const {
  TODO();
}

Maybe<int64_t> JobBuildAndInferCtx::GetSplitDimFromProducerView(const std::string& lbn) const {
  TODO();
}

Maybe<ParallelDesc> JobBuildAndInferCtx::GetParallelDescFromProducerView(
    const std::string& lbn) const {
  TODO();
}

const Job& JobBuildAndInferCtx::job() const { return *job_; }

}  // namespace oneflow
