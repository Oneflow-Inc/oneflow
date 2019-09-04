#include "oneflow/core/job/job_build_and_infer_ctx.h"

namespace oneflow {

std::shared_ptr<ErrorProto> GenJobBuildAndInferError(JobBuildAndInferError err_code,
                                                     std::string msg) {
  auto err = std::make_shared<ErrorProto>();
  err->set_msg(msg);
  err->set_job_build_and_infer_error(err_code);
  return err;
}

JobBuildAndInferCtx::JobBuildAndInferCtx(Job* job, int64_t job_id) : job_(job), job_id_(job_id) {
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
  CHECK_ISNULL(Global<JobDesc>::Get());
  Global<JobDesc>::New(job_conf, job_id_);
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::AddOpNameParallelConf2Placement(
    const std::string& op_name, const ParallelConf& parallel_conf) {
  if (parallel_conf2placement_group_id_.find(parallel_conf)
      == parallel_conf2placement_group_id_.end()) {
    parallel_conf2placement_group_id_.emplace(parallel_conf,
                                              job_->placement().placement_group_size());
    *(job_->mutable_placement()->add_placement_group()->mutable_parallel_conf()) = parallel_conf;
  }
  PlacementGroup* pg = job_->mutable_placement()->mutable_placement_group(
      parallel_conf2placement_group_id_.at(parallel_conf));
  CHECK(pg->parallel_conf() == parallel_conf);
  pg->mutable_op_set()->add_op_name(op_name);
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::DecodeSplitHint7AddOp7AddSbpSigConf2Job(
    Operator* op, SbpSignature* sbp_sig_conf) {
  OperatorConf op_conf_without_split_hint = op->op_conf();
  PbMessage* op_type_conf = MutableMessageInPbMessage(&op_conf_without_split_hint,
                                                      op_conf_without_split_hint.op_type_case());
  bool has_user_set_sbp_sig_conf = false;
  for (const std::string& ibn : op->input_bns()) {
    std::string lbn_may_with_split_hint = GetStrValInPbFdOrPbRpf(op->GetCustomizedConf(), ibn);
    if (HasSplitHintInLbn(lbn_may_with_split_hint)) {
      has_user_set_sbp_sig_conf = true;
      SbpParallel sbp_parallel = GetSbpParallelInLbn(lbn_may_with_split_hint);
      (*(sbp_sig_conf->mutable_bn_in_op2sbp_parallel()))[ibn] = sbp_parallel;
      const LogicalBlobId& lbi = op->BnInOp2Lbi(ibn);
      std::string lbn = GenLogicalBlobName(lbi);
      ReplaceStrValInPbFdOrPbRpf(op_type_conf, ibn, lbn_may_with_split_hint, lbn);
    }
  }
  if (has_user_set_sbp_sig_conf) {
    (*(job_->mutable_sbp_conf()->mutable_op_name2sbp_signature_conf()))[op->op_name()] =
        *sbp_sig_conf;
  }
  job_->mutable_net()->add_op()->CopyFrom(op_conf_without_split_hint);
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::InferOpOutSbpParallel(Operator* op,
                                                       const SbpSignature& sbp_sig_conf,
                                                       const ParallelDesc& parallel_desc,
                                                       SbpSignature* sbp_sig_to_infer) {
  HashMap<std::string, SbpInferHint> ibn2sbp_infer_hint;
  for (const std::string& ibn : op->input_bns()) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(ibn);
    if (lbi2logical_blob_desc_.find(lbi) == lbi2logical_blob_desc_.end()) {
      return GenJobBuildAndInferError(
          JobBuildAndInferError::kLogicalBlobNameNotExist,
          "when infer op_name: " + op->op_name() + "consumed op_name: " + lbi.op_name()
              + " blob_name: " + lbi.blob_name() + " not infer blob desc");
    }
    const BlobDesc* logical_blob_desc = lbi2logical_blob_desc_.at(lbi).get();
    if (lbi2sbp_parallel_from_producer_view_.find(lbi)
        == lbi2sbp_parallel_from_producer_view_.end()) {
      return GenJobBuildAndInferError(
          JobBuildAndInferError::kLogicalBlobNameNotExist,
          "when infer op_name: " + op->op_name() + "consumed op_name: " + lbi.op_name()
              + " blob_name: " + lbi.blob_name() + " not infer split axis");
    }
    const SbpParallel& sbp_parallel = lbi2sbp_parallel_from_producer_view_.at(lbi);
    ibn2sbp_infer_hint.emplace(ibn, SbpInferHint(&parallel_desc, logical_blob_desc, sbp_parallel));
  }
  auto SbpInferHint4Ibn = [&](const std::string& ibn) -> Maybe<const SbpInferHint*> {
    auto it = ibn2sbp_infer_hint.find(ibn);
    if (it == ibn2sbp_infer_hint.end()) {
      return Error::CheckFailed() << "cannot find corresponding SbpInferHint for input_blob_name : "
                                  << ibn;
    }
    return &(it->second);
  };
  std::function<int32_t(const SbpSignature&)> CalcOrderValue4SbpSig;
  if (sbp_sig_conf.bn_in_op2sbp_parallel().empty()) {
    auto OrderValue4HasBatchAxis = [&](const std::string& bn,
                                       const SbpParallel& sbp_parallel) -> int32_t {
      const auto& batch_axis = lbi2batch_axis_.at(op->BnInOp2Lbi(bn));
      return -1
             * (batch_axis.has_value() && sbp_parallel.has_split_parallel()
                && sbp_parallel.split_parallel().axis() == batch_axis.value());
    };
    auto OrderValue4HasNoBatchAxis = [&](const std::string& ibn,
                                         const SbpParallel& sbp_parallel) -> int32_t {
      return -2
             * (lbi2batch_axis_.at(op->BnInOp2Lbi(ibn)).has_value() == false
                && CHECK_JUST(SbpInferHint4Ibn(ibn))->sbp_parallel().has_split_parallel() == false
                && sbp_parallel.has_split_parallel() == false);
    };
    CalcOrderValue4SbpSig = [&](const SbpSignature& sbp_signature) -> int32_t {
      int32_t order_value = 0;
      for (const auto& ibn : op->input_bns()) {
        const auto& sbp_parallel_it = sbp_signature.bn_in_op2sbp_parallel().find(ibn);
        CHECK(sbp_parallel_it != sbp_signature.bn_in_op2sbp_parallel().end());
        order_value += OrderValue4HasBatchAxis(ibn, sbp_parallel_it->second);
        order_value += OrderValue4HasNoBatchAxis(ibn, sbp_parallel_it->second);
      }
      for (const auto& obn : op->output_bns()) {
        const auto& sbp_parallel_it = sbp_signature.bn_in_op2sbp_parallel().find(obn);
        CHECK(sbp_parallel_it != sbp_signature.bn_in_op2sbp_parallel().end());
        order_value += OrderValue4HasBatchAxis(obn, sbp_parallel_it->second);
      }
      return order_value;
    };
  } else {
    CalcOrderValue4SbpSig = [](const SbpSignature&) -> int32_t { return 0; };
  }
  JUST(op->InferSbpSignatureIf(sbp_sig_to_infer, sbp_sig_conf, CalcOrderValue4SbpSig,
                               SbpInferHint4Ibn, parallel_desc));

  const auto& bn2sbp_parallel = sbp_sig_to_infer->bn_in_op2sbp_parallel();
  for (const auto& obn : op->output_bns()) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(obn);
    if (bn2sbp_parallel.find(obn) == bn2sbp_parallel.end()) {
      return GenJobBuildAndInferError(
          JobBuildAndInferError::kBlobSplitAxisInferError,
          "op_name: " + lbi.op_name() + " blob_name: " + lbi.blob_name() + " not infer split axis");
    }
    if (lbi2sbp_parallel_from_producer_view_.emplace(lbi, bn2sbp_parallel.at(obn)).second
        == false) {
      return GenJobBuildAndInferError(JobBuildAndInferError::kBlobSplitAxisInferError,
                                      "op_name: " + lbi.op_name() + " blob_name: " + lbi.blob_name()
                                          + " infer split axis repeated");
    }
    if (lbi2parallel_desc_from_producer_view_.emplace(lbi, parallel_desc).second == false) {
      return GenJobBuildAndInferError(JobBuildAndInferError::kBlobSplitAxisInferError,
                                      "op_name: " + lbi.op_name() + " blob_name: " + lbi.blob_name()
                                          + " add parallel desc repeated");
    }
  }
  return Maybe<void>::Ok();
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
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::CheckOpBlobCanBeSplitedByParallelNum(Operator* op,
                                                                      const SbpSignature& sbp_sig,
                                                                      int64_t parallel_num) {
  for (const auto& pair : sbp_sig.bn_in_op2sbp_parallel()) {
    if (pair.second.has_split_parallel()) {
      int64_t axis = pair.second.split_parallel().axis();
      const LogicalBlobId& lbi = op->BnInOp2Lbi(pair.first);
      const BlobDesc& logical_blob_desc = *(lbi2logical_blob_desc_.at(lbi).get());
      int64_t num_axes = logical_blob_desc.shape().NumAxes();
      if (axis < 0) { axis += num_axes; }
      if (axis < 0 || axis >= num_axes || logical_blob_desc.shape().At(axis) < parallel_num) {
        return GenJobBuildAndInferError(
            JobBuildAndInferError::kUnknownJobBuildAndInferError,
            "op_name: " + lbi.op_name() + " blob_name: " + lbi.blob_name()
                + " cannot split blob by parallel_num: " + std::to_string(parallel_num));
      }
    }
  }
  return Maybe<void>::Ok();
}

// TODO(): add handle error of same interface op blob between jobs
Maybe<void> JobBuildAndInferCtx::AddAndInferOp(const OperatorConf& op_conf,
                                               const ParallelConf& parallel_conf) {
  if (!has_job_conf_) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kJobConfNotSet, "");
  }
  if (!is_job_conf_frozen_) { is_job_conf_frozen_ = true; }

  const std::string& op_name = op_conf.name();
  if (op_name2op_.find(op_name) != op_name2op_.end()) {
    return GenJobBuildAndInferError(
        JobBuildAndInferError::kOpNameExist,
        "op_name: " + op_name + " already exist in job: " + job_->job_conf().job_name());
  }
  if (op_conf.device_type() == DeviceType::kInvalidDevice) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kOpConfDeviceTypeNoSet,
                                    "op_name: " + op_name + " not set device type");
  }

  JUST(AddOpNameParallelConf2Placement(op_name, parallel_conf));

  op_name2op_.emplace(op_name, ConstructOp(op_conf));
  Operator* op = op_name2op_.at(op_name).get();

  SbpSignature sbp_sig_conf;
  JUST(DecodeSplitHint7AddOp7AddSbpSigConf2Job(op, &sbp_sig_conf));

  // infer batch_axis
  auto BatchAxis4BnInOp = [&](const std::string& bn) -> OptInt64* {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(bn);
    return &(lbi2batch_axis_[lbi]);
  };
  auto GetConstBlobDescBnInOp = [&](const std::string& bn) -> const BlobDesc& {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(bn);
    return *(lbi2logical_blob_desc_[lbi].get());
  };
  JUST(op->InferBatchAxisIf(GetConstBlobDescBnInOp, BatchAxis4BnInOp));

  // infer sbp
  ParallelDesc parallel_desc(parallel_conf);
  SbpSignature sbp_sig_to_infer;
  JUST(InferOpOutSbpParallel(op, sbp_sig_conf, parallel_desc, &sbp_sig_to_infer));

  // infer logical blob desc
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
  JUST(op->InferOutBlobDescsIf(GetBlobDesc4BnInOp, &parallel_ctx, job_->job_conf().piece_size(),
                               [](OpContext*) {}));

  // check blob can be split
  JUST(CheckOpBlobCanBeSplitedByParallelNum(op, sbp_sig_to_infer, parallel_desc.parallel_num()));

  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::AddLossLogicalBlobName(const std::string& lbn) {
  if (!(job_->job_conf().has_train_conf())) {
    return GenJobBuildAndInferError(JobBuildAndInferError::kUnknownJobBuildAndInferError,
                                    "job has not TrainConf when add loss logical blob name");
  }
  job_->mutable_job_conf()->mutable_train_conf()->add_loss_lbn(lbn);
  return Maybe<void>::Ok();
}

bool JobBuildAndInferCtx::HasJobConf() const { return has_job_conf_; }

Maybe<void> JobBuildAndInferCtx::AddPlacementGroup(const PlacementGroup& placement_group) {
  // OUTDATE need to be deleted
  job_->mutable_placement()->add_placement_group()->CopyFrom(placement_group);
  return Maybe<void>::Ok();
}

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

Maybe<OptInt64> JobBuildAndInferCtx::GetBatchAxis(const std::string& lbn) const {
  GEN_ERROR_WHEN_GET_INFO_FROM_LBN(lbi2batch_axis_);
  return lbi2batch_axis_.at(lbi);
}

Maybe<bool> JobBuildAndInferCtx::GetHasSplitAxisFromProducerView(const std::string& lbn) const {
  GEN_ERROR_WHEN_GET_INFO_FROM_LBN(lbi2sbp_parallel_from_producer_view_);
  return lbi2sbp_parallel_from_producer_view_.at(lbi).has_split_parallel();
}

Maybe<int64_t> JobBuildAndInferCtx::GetSplitAxisFromProducerView(const std::string& lbn) const {
  GEN_ERROR_WHEN_GET_INFO_FROM_LBN(lbi2sbp_parallel_from_producer_view_);
  const SbpParallel& sbp = lbi2sbp_parallel_from_producer_view_.at(lbi);
  if (sbp.has_split_parallel()) {
    return sbp.split_parallel().axis();
  } else {
    return GenJobBuildAndInferError(JobBuildAndInferError::kLogicalBlobNameInvalid,
                                    "lbn:" + lbn + " has no split axis from producer view ");
  }
}

Maybe<ParallelDesc> JobBuildAndInferCtx::GetParallelDescFromProducerView(
    const std::string& lbn) const {
  GEN_ERROR_WHEN_GET_INFO_FROM_LBN(lbi2parallel_desc_from_producer_view_);
  return lbi2parallel_desc_from_producer_view_.at(lbi);
}

Maybe<void> JobBuildAndInferCtx::CheckJob() const {
  JUST(CheckPlacement());
  JUST(CheckJobConf());
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::CheckPlacement() const {
  HashSet<std::string> op_names_in_net;
  HashSet<std::string> op_names_in_placement;
  for (const OperatorConf& op_conf : job_->net().op()) {
    if (!(op_names_in_net.insert(op_conf.name()).second)) {
      return GenJobBuildAndInferError(JobBuildAndInferError::kOpNameExist,
                                      "op_name: " + op_conf.name() + " already exist in job: "
                                          + job_->job_conf().job_name() + " net");
    }
  }
  for (const PlacementGroup& placement_group : job_->placement().placement_group()) {
    for (const std::string& op_name : placement_group.op_set().op_name()) {
      if (!(op_names_in_placement.insert(op_name).second)) {
        return GenJobBuildAndInferError(JobBuildAndInferError::kOpNameExist,
                                        "op_name: " + op_name + " already exist in job: "
                                            + job_->job_conf().job_name() + " placement");
      }
    }
  }
  if (op_names_in_net.size() != op_names_in_placement.size()) {
    return GenJobBuildAndInferError(
        JobBuildAndInferError::kPlacementError,
        "job: " + job_->job_conf().job_name() + " op number not equal between net and placement");
  }
  for (const std::string& op_name : op_names_in_net) {
    if (op_names_in_placement.find(op_name) == op_names_in_placement.end()) {
      return GenJobBuildAndInferError(JobBuildAndInferError::kPlacementError,
                                      "job: " + job_->job_conf().job_name() + " op_name: " + op_name
                                          + " defined in net cannot find its placement");
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::CheckJobConf() const {
  if (job_->job_conf().job_type_case() == JobConfigProto::JOB_TYPE_NOT_SET) {
    return Error::JobTypeNotSet() << "job_type not set, please set predict_conf or train_conf";
  }
  return Maybe<void>::Ok();
}

const Job& JobBuildAndInferCtx::job() const { return *job_; }

}  // namespace oneflow
