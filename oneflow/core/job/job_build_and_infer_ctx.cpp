#include "oneflow/core/job/job_build_and_infer_ctx.h"

namespace oneflow {

JobBuildAndInferCtx::JobBuildAndInferCtx(Job* job, int64_t job_id) : job_(job), job_id_(job_id) {
  is_job_conf_frozen_ = false;
  has_job_conf_ = false;
}

Maybe<void> JobBuildAndInferCtx::SetJobConf(const JobConfigProto& job_conf) {
  CHECK_OR_RETURN(!is_job_conf_frozen_) << JobBuildAndInferError::kJobConfFrozen;
  CHECK_OR_RETURN(!has_job_conf_) << JobBuildAndInferError::kJobConfRepeatedSet;
  has_job_conf_ = true;
  CHECK_EQ_OR_RETURN(job_->job_conf().job_name(), job_conf.job_name())
      << JobBuildAndInferError::kJobNameNotEqual << "job name you set: " << job_conf.job_name()
      << " not equal to origin job name: " << job_->job_conf().job_name();
  job_->mutable_job_conf()->CopyFrom(job_conf);
  CHECK_ISNULL(Global<JobDesc>::Get());
  Global<JobDesc>::New(job_conf, job_id_);
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::AddOpNameParallelConf2Placement(
    const std::string& op_name, const ParallelConf& parallel_conf) {
  ParallelDesc parallel_desc(parallel_conf);
  PlacementGroup* pg = nullptr;
  if (parallel_desc2placement_group_.find(parallel_desc) == parallel_desc2placement_group_.end()) {
    pg = job_->mutable_placement()->add_placement_group();
    parallel_desc2placement_group_.emplace(parallel_desc, pg);
    *(pg->mutable_parallel_conf()) = parallel_conf;
  } else {
    pg = parallel_desc2placement_group_.at(parallel_desc);
  }
  pg->mutable_op_set()->add_op_name(op_name);
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::AddLbiParallelConf2BlobPlacement(
    const Operator* op, std::function<ParallelDesc*(const std::string&)> ParallelDesc4Obn) {
  for (const auto& obn : op->output_bns()) {
    const auto& parallel_desc = *ParallelDesc4Obn(obn);
    auto iter = parallel_desc2blob_placement_group_.find(parallel_desc);
    if (iter == parallel_desc2blob_placement_group_.end()) {
      auto* blob_pg = job_->mutable_placement()->add_blob_placement_group();
      *blob_pg->mutable_parallel_conf() = parallel_desc.parallel_conf();
      iter = parallel_desc2blob_placement_group_.emplace(parallel_desc, blob_pg).first;
    }
    const auto& lbi = op->BnInOp2Lbi(obn);
    OF_CHECK(std::find(iter->second->lbi().begin(), iter->second->lbi().end(), lbi)
             == iter->second->lbi().end());
    *iter->second->add_lbi() = lbi;
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::DecodeSplitHint7AddOp7AddSbpSigConf2Job(
    Operator* op, SbpSignature* sbp_sig_conf) {
  OperatorConf op_conf_without_split_hint = op->op_conf();
  PbMessage* op_type_conf = MutableMessageInPbMessage(&op_conf_without_split_hint,
                                                      op_conf_without_split_hint.op_type_case());
  for (const std::string& ibn : op->input_bns()) {
    std::string lbn_may_with_split_hint = GetStrValInPbFdOrPbRpf(op->GetCustomizedConf(), ibn);
    SbpParallel sbp_parallel;
    if (JUST(GetSbpParallelInLbnOrNothing(lbn_may_with_split_hint, &sbp_parallel))) {
      (*(sbp_sig_conf->mutable_bn_in_op2sbp_parallel()))[ibn] = sbp_parallel;
      const LogicalBlobId& lbi = op->BnInOp2Lbi(ibn);
      std::string lbn = GenLogicalBlobName(lbi);
      ReplaceStrValInPbFdOrPbRpf(op_type_conf, ibn, lbn_may_with_split_hint, lbn);
    }
  }
  if (sbp_sig_conf->bn_in_op2sbp_parallel().size() > 0) {
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
    CHECK_OR_RETURN(lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end())
        << JobBuildAndInferError::kLogicalBlobNameNotExist << "when infer op_name: \""
        << op->op_name() << "\", consumed op_name: \"" << lbi.op_name() << "\", blob_name: \""
        << lbi.blob_name() << "\" not infer blob desc";
    const ParallelDesc* pd = &lbi2parallel_desc_from_producer_view_.at(lbi);
    const BlobDesc* logical_blob_desc = lbi2logical_blob_desc_.at(lbi).get();
    CHECK_OR_RETURN(lbi2sbp_parallel_from_producer_view_.find(lbi)
                    != lbi2sbp_parallel_from_producer_view_.end())
        << JobBuildAndInferError::kLogicalBlobNameNotExist
        << "when infer op_name: " << op->op_name() << " consumed op_name: " << lbi.op_name()
        << " blob_name: " << lbi.blob_name() << " not infer split axis";
    const SbpParallel& sbp_parallel = lbi2sbp_parallel_from_producer_view_.at(lbi);
    ibn2sbp_infer_hint.emplace(ibn, SbpInferHint(pd, logical_blob_desc, sbp_parallel));
  }

  auto GetBatchAxis4Lbi = [&](const LogicalBlobId& lbi) -> const OptInt64& {
    return lbi2batch_axis_.at(lbi);
  };

  CHECK_JUST(InferOpSbpSignature(*op, sbp_sig_conf, parallel_desc, ibn2sbp_infer_hint,
                                 GetBatchAxis4Lbi, sbp_sig_to_infer));

  const auto& bn2sbp_parallel = sbp_sig_to_infer->bn_in_op2sbp_parallel();
  for (const auto& obn : op->output_bns()) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(obn);
    CHECK_OR_RETURN(bn2sbp_parallel.find(obn) != bn2sbp_parallel.end())
        << JobBuildAndInferError::kBlobSplitAxisInferError << "op_name: " << lbi.op_name()
        << " blob_name: " << lbi.blob_name() << " not infer split axis";
    CHECK_OR_RETURN(
        lbi2sbp_parallel_from_producer_view_.emplace(lbi, bn2sbp_parallel.at(obn)).second)
        << JobBuildAndInferError::kBlobSplitAxisInferError << "op_name: " << lbi.op_name()
        << " blob_name: " << lbi.blob_name() << " infer split axis repeated";
    CHECK_OR_RETURN(lbi2parallel_desc_from_producer_view_.emplace(lbi, parallel_desc).second)
        << JobBuildAndInferError::kBlobSplitAxisInferError << "op_name: " << lbi.op_name()
        << " blob_name: " << lbi.blob_name() << " add parallel desc repeated";
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::GenOpProducedEmptyLogicalBlobDesc(Operator* op) {
  // check consumed blob
  for (const std::string& consumed_bn : op->input_bns()) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(consumed_bn);
    CHECK_OR_RETURN(lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end())
        << JobBuildAndInferError::kLogicalBlobNameNotExist << "op_name: " << op->op_name()
        << " consumed_op_name:" << lbi.op_name() << " blob_name: " << lbi.blob_name()
        << " not exist";
  }

  // create produced blob
  std::vector<std::string> produced_bns;
  produced_bns.insert(produced_bns.end(), op->output_bns().begin(), op->output_bns().end());
  produced_bns.insert(produced_bns.end(), op->tmp_bns().begin(), op->tmp_bns().end());
  produced_bns.insert(produced_bns.end(), op->const_buf_bns().begin(), op->const_buf_bns().end());
  for (const std::string& produced_bn : produced_bns) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(produced_bn);
    CHECK_OR_RETURN(lbi2logical_blob_desc_.find(lbi) == lbi2logical_blob_desc_.end())
        << JobBuildAndInferError::kLogicalBlobNameRepeated << "op_name: " << lbi.op_name()
        << " blob_name: " << lbi.blob_name() << " is repeated";
    lbi2logical_blob_desc_.emplace(lbi, std::make_unique<BlobDesc>(DataType::kInvalidDataType));
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::CheckOpBlobSplitability(Operator* op, const SbpSignature& sbp_sig,
                                                         int64_t parallel_num) {
  HashSet<std::string> obns(op->output_bns().begin(), op->output_bns().end());
  auto GetParallelNum = [&](const std::string& bn_in_op) {
    if (obns.find(bn_in_op) == obns.end()) { return parallel_num; }
    return lbi2parallel_desc_from_producer_view_.at(op->BnInOp2Lbi(bn_in_op)).parallel_num();
  };
  for (const auto& pair : sbp_sig.bn_in_op2sbp_parallel()) {
    if (pair.second.has_split_parallel()) {
      int64_t axis = pair.second.split_parallel().axis();
      const LogicalBlobId& lbi = op->BnInOp2Lbi(pair.first);
      int64_t blob_parallel_num = GetParallelNum(pair.first);
      const BlobDesc& logical_blob_desc = *(lbi2logical_blob_desc_.at(lbi).get());
      int64_t num_axes = logical_blob_desc.shape().NumAxes();
      if (axis < 0) { axis += num_axes; }
      CHECK_OR_RETURN(axis >= 0 && axis < num_axes
                      && logical_blob_desc.shape().At(axis) >= blob_parallel_num)
          << JobBuildAndInferError::kUnknownJobBuildAndInferError << "op_name: " << lbi.op_name()
          << " blob_name: " << lbi.blob_name()
          << " cannot split blob by parallel_num: " << std::to_string(blob_parallel_num);
    }
  }
  return Maybe<void>::Ok();
}

// TODO(): add handle error of same interface op blob between jobs
Maybe<void> JobBuildAndInferCtx::AddAndInferOp(const OperatorConf& op_conf,
                                               const ParallelConf& parallel_conf) {
  CHECK_OR_RETURN(has_job_conf_) << JobBuildAndInferError::kJobConfNotSet;
  if (!is_job_conf_frozen_) { is_job_conf_frozen_ = true; }
  const std::string& op_name = op_conf.name();
  CHECK_OR_RETURN(op_name2op_.find(op_name) == op_name2op_.end())
      << JobBuildAndInferError::kOpNameExist << "op_name: " << op_name
      << " already exist in job: " << job_->job_conf().job_name();
  CHECK_NE_OR_RETURN(op_conf.device_type(), DeviceType::kInvalidDevice)
      << JobBuildAndInferError::kOpConfDeviceTypeNoSet << "op_name: " << op_name
      << " not set device type";

  JUST(AddOpNameParallelConf2Placement(op_name, parallel_conf));

  op_name2op_.emplace(op_name, ConstructOp(op_conf, &GlobalJobDesc()));
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
  JUST(op->InferOutBlobDescsIf(GetBlobDesc4BnInOp, &parallel_ctx, &sbp_sig_to_infer,
                               [](OpContext*) {}));
  auto ParallelDesc4Obn = [&](const std::string& obn) -> ParallelDesc* {
    const auto& lbi = op->BnInOp2Lbi(obn);
    auto iter = lbi2parallel_desc_from_producer_view_.find(lbi);
    if (iter == lbi2parallel_desc_from_producer_view_.end()) {
      iter = lbi2parallel_desc_from_producer_view_.emplace(lbi, parallel_desc).first;
    }
    return &iter->second;
  };
  JUST(op->InferOutParallelDescIf(ParallelDesc4Obn, GetBlobDesc4BnInOp, parallel_desc,
                                  &sbp_sig_to_infer));
  JUST(AddLbiParallelConf2BlobPlacement(op, ParallelDesc4Obn));
  // check splitability
  JUST(CheckOpBlobSplitability(op, sbp_sig_to_infer, parallel_desc.parallel_num()));

  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::AddLossLogicalBlobName(const std::string& lbn) {
  CHECK_OR_RETURN(job_->job_conf().has_train_conf())
      << JobBuildAndInferError::kUnknownJobBuildAndInferError
      << "job has not TrainConf when add loss logical blob name";
  job_->mutable_job_conf()->mutable_train_conf()->add_loss_lbn(lbn);
  return Maybe<void>::Ok();
}

bool JobBuildAndInferCtx::HasJobConf() const { return has_job_conf_; }

Maybe<Shape> JobBuildAndInferCtx::GetStaticShape(const std::string& lbn) const {
  JUST(CheckLbnValidAndExist(lbn));
  return lbi2logical_blob_desc_.at(GenLogicalBlobId(lbn))->shape();
}

Maybe<DataType> JobBuildAndInferCtx::GetDataType(const std::string& lbn) const {
  JUST(CheckLbnValidAndExist(lbn));
  return lbi2logical_blob_desc_.at(GenLogicalBlobId(lbn))->data_type();
}

Maybe<bool> JobBuildAndInferCtx::IsDynamic(const std::string& lbn) const {
  JUST(CheckLbnValidAndExist(lbn));
  return lbi2logical_blob_desc_.at(GenLogicalBlobId(lbn))->is_dynamic();
}

Maybe<long long> JobBuildAndInferCtx::GetNumOfLoDLevels(const std::string& lbn) const {
  JUST(CheckLbnValidAndExist(lbn));
  return lbi2logical_blob_desc_.at(GenLogicalBlobId(lbn))->num_of_lod_levels();
}

Maybe<OptInt64> JobBuildAndInferCtx::GetBatchAxis(const std::string& lbn) const {
  JUST(CheckLbnValidAndExist(lbn));
  return lbi2batch_axis_.at(GenLogicalBlobId(lbn));
}

Maybe<OptInt64> JobBuildAndInferCtx::GetSplitAxisFromProducerView(const std::string& lbn) const {
  JUST(CheckLbnValidAndExist(lbn));
  OptInt64 ret;
  const auto& sbp = lbi2sbp_parallel_from_producer_view_.at(GenLogicalBlobId(lbn));
  if (sbp.has_split_parallel()) { ret.set_value(sbp.split_parallel().axis()); }
  return ret;
}

Maybe<const ParallelDesc*> JobBuildAndInferCtx::GetParallelDescFromProducerView(
    const std::string& lbn) const {
  JUST(CheckLbnValidAndExist(lbn));
  return &(lbi2parallel_desc_from_producer_view_.at(GenLogicalBlobId(lbn)));
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
    CHECK_OR_RETURN(op_names_in_net.insert(op_conf.name()).second)
        << JobBuildAndInferError::kOpNameExist << "op_name: " << op_conf.name()
        << " already exist in job: " << job_->job_conf().job_name() << " net";
  }
  for (const PlacementGroup& placement_group : job_->placement().placement_group()) {
    for (const std::string& op_name : placement_group.op_set().op_name()) {
      CHECK_OR_RETURN(op_names_in_placement.insert(op_name).second)
          << JobBuildAndInferError::kOpNameExist << "op_name: " << op_name
          << " already exist in job: " << job_->job_conf().job_name() << " placement";
    }
  }
  CHECK_EQ_OR_RETURN(op_names_in_net.size(), op_names_in_placement.size())
      << JobBuildAndInferError::kPlacementError << "job: " << job_->job_conf().job_name()
      << " op number not equal between net and placement";
  for (const std::string& op_name : op_names_in_net) {
    CHECK_OR_RETURN(op_names_in_placement.find(op_name) != op_names_in_placement.end())
        << JobBuildAndInferError::kPlacementError << "job: " << job_->job_conf().job_name()
        << " op_name: " << op_name << " defined in net cannot find its placement";
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::CheckJobConf() const {
  if (job_->job_conf().job_type_case() == JobConfigProto::JOB_TYPE_NOT_SET) {
    return Error::JobTypeNotSet() << "job_type not set, please set predict_conf or train_conf";
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::CheckLbnValidAndExist(const std::string& lbn) const {
  CHECK_OR_RETURN(lbn.find('/') != std::string::npos)
      << JobBuildAndInferError::kLogicalBlobNameInvalid << "lbn:" << lbn;
  LogicalBlobId lbi = GenLogicalBlobId(lbn);

#define CHECK_HAS_LBI_KEY(info_src)                     \
  CHECK_OR_RETURN(info_src.find(lbi) != info_src.end()) \
      << JobBuildAndInferError::kLogicalBlobNameNotExist << "lbn:" << lbn;

  CHECK_HAS_LBI_KEY(lbi2logical_blob_desc_);
  CHECK_HAS_LBI_KEY(lbi2sbp_parallel_from_producer_view_);
  CHECK_HAS_LBI_KEY(lbi2batch_axis_);
  CHECK_HAS_LBI_KEY(lbi2parallel_desc_from_producer_view_);
#undef CHECK_HAS_LBI_KEY

  return Maybe<void>::Ok();
}

const Job& JobBuildAndInferCtx::job() const { return *job_; }

}  // namespace oneflow
