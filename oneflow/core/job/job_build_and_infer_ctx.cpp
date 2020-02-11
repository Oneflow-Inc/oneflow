#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

static const std::string kAutoMirroredBlobNamePrefix =
    "System-Mirrored-Blob-Auto-Converted-From-Consistent-Blob";

namespace {

void ResetOpConfIbn(OperatorConf* op_conf, const std::string& ibn, const std::string lbn) {
  PbMessage* op_type_conf = MutableMessageInPbMessage(op_conf, op_conf->op_type_case());
  std::string lbn_may_with_hint = GetInputLbnInOpCustomizedConf(*op_type_conf, ibn);
  ReplaceInputLbnInOpCustomizedConf(op_type_conf, ibn, lbn_may_with_hint, lbn);
}

void ResetOpConfName(OperatorConf* op_conf, const std::string& new_op_name) {
  op_conf->set_name(new_op_name);
  PbMessage* op_type_conf = MutableMessageInPbMessage(op_conf, op_conf->op_type_case());
  UserOpConf* user_conf = dynamic_cast<UserOpConf*>(op_type_conf);
  if (user_conf) {
    for (const auto& pair : user_conf->output()) {
      for (const std::string& old_lbn : pair.second.s()) {
        LogicalBlobId old_lbi = GenLogicalBlobId(old_lbn);
        auto blob_name_id_pair = GenUnRepeatedBn(old_lbi.blob_name());
        std::string new_lbn = GenLogicalBlobName(new_op_name, old_lbi.blob_name());
        (*(user_conf->mutable_output()))[pair.first].set_s(blob_name_id_pair.second, new_lbn);
      }
    }
  }
}

}  // namespace

JobBuildAndInferCtx::JobBuildAndInferCtx(Job* job, int64_t job_id) : job_(job), job_id_(job_id) {
  is_job_conf_frozen_ = false;
  has_job_conf_ = false;
}

Maybe<OperatorConf> JobBuildAndInferCtx::CheckAndCompleteUserOpConf(const OperatorConf& op_conf) {
  return CheckAndCompleteUserOpConfImpl(op_conf);
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

Maybe<void> JobBuildAndInferCtx::Complete() {
  CHECK_NOTNULL(Global<JobDesc>::Get());
  Global<JobDesc>::Delete();
  auto scope = std::make_unique<GlobalJobDescScope>(job_->job_conf(), job_id_);
  auto DoPass = [&](const std::string& pass_name) { FunctionPass(pass_name)(job_); };
  if (GlobalJobDesc().Bool("__is_user_function__")) {
    DoPass("CompleteOfrecordDecoder");
    DoPass("SetDefaultVariableConf");
    DoPass("AutoMixedPrecision");
    DoPass("TieUpChainHeadersUnReachableFromAnyVariableOps");
    DoPass("NonDistributedOptimizerPass");
    DoPass("AutoTrainStep");
    DoPass("AutoLearningRate");
    DoPass("GenerateBackwardAndOptimizerOpConfs");
    DoPass("SequentializeNcclTupleBroadcastReducePass");
    DoPass("AddAllReduceGroupPass");
    DoPass("AddLbiDiffWatcherOpConfs");
    DoPass("SequentializeAllReduceGroupPass");
    DoPass("PruneParallelCastOpsPass");
  }
  DoPass("DumpTimeShapeAndBlobParallelConfPass");
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

Maybe<OperatorConf> JobBuildAndInferCtx::DecodeLbiHintAndReturnNewOpConf(
    const Operator& op, SbpSignature* sbp_sig_conf,
    HashMap<std::string, bool>* ibn2disable_boxing) const {
  auto op_conf_without_split_hint = std::make_shared<OperatorConf>(op.op_conf());
  PbMessage* op_type_conf = MutableMessageInPbMessage(op_conf_without_split_hint.get(),
                                                      op_conf_without_split_hint->op_type_case());
  for (const std::string& ibn : op.input_bns()) {
    std::string lbn_may_with_hint = GetInputLbnInOpCustomizedConf(op.GetCustomizedConf(), ibn);
    SbpParallel sbp_parallel;
    bool has_sbp_hint = JUST(GetSbpParallelInLbnOrNothing(lbn_may_with_hint, &sbp_parallel));
    bool has_disable_boxing_hint =
        JUST(ParseDisableBoxingFlag(lbn_may_with_hint, &(*ibn2disable_boxing)[ibn]));
    if (has_sbp_hint || has_disable_boxing_hint) {
      (*(sbp_sig_conf->mutable_bn_in_op2sbp_parallel()))[ibn] = sbp_parallel;
      const LogicalBlobId& lbi = op.BnInOp2Lbi(ibn);
      std::string lbn = GenLogicalBlobName(lbi);
      ReplaceInputLbnInOpCustomizedConf(op_type_conf, ibn, lbn_may_with_hint, lbn);
    }
  }
  return op_conf_without_split_hint;
}

void JobBuildAndInferCtx::AddOp7AddSbpSigConf2Job(const OperatorConf& operator_conf,
                                                  const SbpSignature& sbp_signature) const {
  auto* op_name2sbp_sig = job_->mutable_sbp_conf()->mutable_op_name2sbp_signature_conf();
  if (sbp_signature.bn_in_op2sbp_parallel().size() > 0) {
    (*op_name2sbp_sig)[operator_conf.name()] = sbp_signature;
  }
  job_->mutable_net()->add_op()->CopyFrom(operator_conf);
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
    const SbpParallel* sbp_parallel = &lbi2sbp_parallel_from_producer_view_.at(lbi);
    const OptInt64* batch_axis = &lbi2batch_axis_.at(lbi);
    ibn2sbp_infer_hint.emplace(ibn, SbpInferHint(pd, logical_blob_desc, sbp_parallel, batch_axis));
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
      OF_CHECK_GE(axis, 0);
      OF_CHECK_LT(axis, num_axes);
      OF_CHECK_GE(logical_blob_desc.shape().At(axis), blob_parallel_num)
          << "op_name: " << lbi.op_name() << " blob_name: " << lbi.blob_name()
          << " cannot split blob by parallel_num: " << std::to_string(blob_parallel_num);
    }
  }
  return Maybe<void>::Ok();
}

Maybe<ParallelConf> JobBuildAndInferCtx::InferOpParallelConf(
    const Operator& op, const ParallelConf& origin_parallel_conf,
    const HashMap<std::string, bool>& ibn2disable_boxing) const {
  const ParallelDesc* parallel_desc = nullptr;
  for (const auto& ibn : op.input_bns()) {
    if (ibn2disable_boxing.at(ibn) == false) { continue; }
    const auto& lbi = op.BnInOp2Lbi(ibn);
    const auto& ibn_parallel_desc = lbi2parallel_desc_from_producer_view_.at(lbi);
    if (parallel_desc == nullptr) {
      parallel_desc = &ibn_parallel_desc;
    } else {
      OF_CHECK_EQ(parallel_desc->parallel_num(), ibn_parallel_desc.parallel_num());
    }
  }
  if (parallel_desc == nullptr) { return std::make_shared<ParallelConf>(origin_parallel_conf); }
  return std::make_shared<ParallelConf>(parallel_desc->parallel_conf());
}

void JobBuildAndInferCtx::InitIbn2DisableBoxing(const Operator& op,
                                                HashMap<std::string, bool>* ibn2disable_boxing) {
  for (const auto& ibn : op.input_bns()) {
    (*ibn2disable_boxing)[ibn] = lbi2disable_boxing_[op.BnInOp2Lbi(ibn)];
  }
}

void JobBuildAndInferCtx::UpdateLbi2DisableBoxing(
    const Operator& op, const HashMap<std::string, bool>& ibn2disable_boxing) {
  bool disable_boxing = false;
  for (const auto& ibn : op.input_bns()) {
    if (ibn2disable_boxing.at(ibn)) {
      disable_boxing = true;
      break;
    }
  }
  for (const auto& obn : op.output_bns()) {
    lbi2disable_boxing_[op.BnInOp2Lbi(obn)] = disable_boxing;
  }
}

bool JobBuildAndInferCtx::HasAnyMirroredBlobInput(const Operator& op) const {
  for (const auto& ibn : op.input_bns()) {
    const auto& lbi = op.BnInOp2Lbi(ibn);
    if (mirrored_lbi2sub_lbis_.find(lbi) != mirrored_lbi2sub_lbis_.end()) { return true; }
  }
  return false;
}

Maybe<const SbpParallel*> JobBuildAndInferCtx::SbpParallel4Lbi(const LogicalBlobId& lbi) const {
  const auto& iter = lbi2sbp_parallel_from_producer_view_.find(lbi);
  OF_CHECK(iter != lbi2sbp_parallel_from_producer_view_.end())
      << "lbn: " << GenLogicalBlobName(lbi) << " undefined";
  return &iter->second;
}

Maybe<const ParallelDesc*> JobBuildAndInferCtx::ParallelDesc4Lbi(const LogicalBlobId& lbi) const {
  const auto& iter = lbi2parallel_desc_from_producer_view_.find(lbi);
  OF_CHECK(iter != lbi2parallel_desc_from_producer_view_.end())
      << "lbn: " << GenLogicalBlobName(lbi) << " undefined";
  return &iter->second;
}

Maybe<bool> JobBuildAndInferCtx::AllInputsBroadcastParallel(const Operator& op) const {
  for (const auto& ibn : op.input_bns()) {
    const LogicalBlobId& lbi = op.BnInOp2Lbi(ibn);
    const auto& iter = mirrored_lbi2sbp_parallel_.find(lbi);
    if (iter != mirrored_lbi2sbp_parallel_.end()) {
      if (!iter->second.has_broadcast_parallel()) { return false; }
    } else {
      if (!JUST(SbpParallel4Lbi(lbi))->has_broadcast_parallel()) { return false; }
    }
  }
  return true;
}

Maybe<void> JobBuildAndInferCtx::CheckAllInputsConvertableToMirroredBlob(const Operator& op) const {
  for (const auto& ibn : op.input_bns()) {
    const auto& lbi = op.BnInOp2Lbi(ibn);
    if (mirrored_lbi2sub_lbis_.find(lbi) != mirrored_lbi2sub_lbis_.end()) { continue; }
    const auto& sbp = *JUST(SbpParallel4Lbi(lbi));
    if (sbp.has_broadcast_parallel()) { continue; }
    if (sbp.has_split_parallel() && sbp.split_parallel().axis() == 0) { continue; }
    const std::string& lbn = GenLogicalBlobName(lbi);
    return Error::CheckFailed() << "input lbn: " << lbn << " is not convertable to mirrored blob";
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::CheckAllInputsWithSameParallelNum(const Operator& op,
                                                                   int32_t parallel_num) const {
  for (const auto& ibn : op.input_bns()) {
    const auto& lbi = op.BnInOp2Lbi(ibn);
    const auto& iter = mirrored_lbi2sub_lbis_.find(lbi);
    int32_t ibn_parallel_num = 0;
    if (iter != mirrored_lbi2sub_lbis_.end()) {
      ibn_parallel_num = iter->second.size();
    } else {
      ibn_parallel_num = JUST(ParallelDesc4Lbi(lbi))->parallel_num();
    }
    OF_CHECK_EQ(ibn_parallel_num, parallel_num)
        << "the parallel_num of input lbn: " << GenLogicalBlobName(lbi)
        << "is not equals to op' parallel_num";
  }
  return Maybe<void>::Ok();
}

Maybe<LogicalBlobId> JobBuildAndInferCtx::FindOrCreateMirroredLbiFromCompatibleConsistentBlob(
    const LogicalBlobId& lbi) {
  const std::string& lbn = GenLogicalBlobName(lbi);
  const auto& sbn_it = consistent_lbi2mirrored_lbi_.find(lbi);
  if (sbn_it != consistent_lbi2mirrored_lbi_.end()) { return sbn_it->second; }
  const SbpParallel& sbp = *CHECK_JUST(SbpParallel4Lbi(lbi));
  const ParallelDesc& parallel_desc = *CHECK_JUST(ParallelDesc4Lbi(lbi));
  LogicalBlobId mirrored_lbi;
  mirrored_lbi.set_op_name(kAutoMirroredBlobNamePrefix + NewUniqueId());
  mirrored_lbi.set_blob_name("out");
  consistent_lbi2mirrored_lbi_[lbi] = mirrored_lbi;
  auto* lbi_vec = &mirrored_lbi2sub_lbis_[mirrored_lbi];
  lbi_vec->reserve(parallel_desc.parallel_num());
  auto PushBackSubLbi = [&](const std::string& op_name, const std::string& blob_name) {
    LogicalBlobId sub_lbi;
    sub_lbi.set_op_name(op_name);
    sub_lbi.set_blob_name(blob_name);
    lbi_vec->push_back(sub_lbi);
  };
  OperatorConf op_conf;
  op_conf.set_device_type(parallel_desc.device_type());
  if (sbp.has_broadcast_parallel()) {
    op_conf.set_name(kAutoMirroredBlobNamePrefix + "-DistributeClone-" + NewUniqueId());
    auto* distribute_clone = op_conf.mutable_distribute_clone_conf();
    distribute_clone->set_in(lbn);
    FOR_RANGE(int32_t, i, 0, parallel_desc.parallel_num()) {
      const std::string& blob_name = "out_" + std::to_string(i);
      distribute_clone->add_out(blob_name);
      PushBackSubLbi(op_conf.name(), blob_name);
    }
  } else if (sbp.has_split_parallel()) {
    OF_CHECK_EQ(sbp.split_parallel().axis(), 0)
        << "only `S(0)' consistent blob is compatible to mirrored blob";
    op_conf.set_name(kAutoMirroredBlobNamePrefix + "-DistributeSplit-" + NewUniqueId());
    auto* distribute_split = op_conf.mutable_distribute_split_conf();
    distribute_split->set_in(lbn);
    distribute_split->set_axis(0);
    FOR_RANGE(int32_t, i, 0, parallel_desc.parallel_num()) {
      const std::string& blob_name = "out_" + std::to_string(i);
      distribute_split->add_out(blob_name);
      PushBackSubLbi(op_conf.name(), blob_name);
    }
  } else {
    OF_UNIMPLEMENTED() << "`P' consistant blob is not compatible to mirrored blob";
  }
  CHECK_JUST(AddAndInferConsistentOp(op_conf, parallel_desc.parallel_conf()));
  return mirrored_lbi;
}

Maybe<void> JobBuildAndInferCtx::AddAndInferOp(const OperatorConf& op_conf,
                                               const ParallelConf& origin_parallel_conf) {
  auto op = ConstructOp(op_conf, &GlobalJobDesc());
  if (HasAnyMirroredBlobInput(*op)) { return AddAndInferMirroredOp(op_conf, origin_parallel_conf); }
  return AddAndInferConsistentOp(op_conf, origin_parallel_conf);
}

Maybe<void> JobBuildAndInferCtx::AddAndInferMirroredOp(const OperatorConf& op_conf,
                                                       const ParallelConf& origin_parallel_conf) {
  auto op = ConstructOp(op_conf, &GlobalJobDesc());
  JUST(CheckAllInputsConvertableToMirroredBlob(*op));
  ParallelDesc parallel_desc(origin_parallel_conf);
  int32_t parallel_num = parallel_desc.parallel_num();
  JUST(CheckAllInputsWithSameParallelNum(*op, parallel_num));
  auto GetSubOpName = [&](int index) { return op_conf.name() + "_" + std::to_string(index); };
  OperatorConf sub_op_conf(op_conf);
  FOR_RANGE(int32_t, i, 0, parallel_num) {
    ResetOpConfName(&sub_op_conf, GetSubOpName(i));
    for (const auto& ibn : op->input_bns()) {
      const auto& lbi = *JUST(GetSubLbi(op->BnInOp2Lbi(ibn), i));
      ResetOpConfIbn(&sub_op_conf, ibn, GenLogicalBlobName(lbi));
    }
    AddAndInferConsistentOp(sub_op_conf, parallel_desc.GetParallelIdOnlyParallelConf(i));
  }
  bool is_broadcast = JUST(AllInputsBroadcastParallel(*op));
  for (const auto& obn : op->output_bns()) {
    const auto& lbi = op->BnInOp2Lbi(obn);
    auto* sub_lbis = &mirrored_lbi2sub_lbis_[lbi];
    sub_lbis->resize(parallel_num, op->BnInOp2Lbi(obn));
    FOR_RANGE(int32_t, i, 0, parallel_num) { sub_lbis->at(i).set_op_name(GetSubOpName(i)); }
    CHECK(mirrored_lbi2parallel_desc_.emplace(lbi, parallel_desc).second);
    auto* sbp_parallel = &mirrored_lbi2sbp_parallel_[lbi];
    if (is_broadcast) {
      sbp_parallel->mutable_broadcast_parallel();
    } else {
      sbp_parallel->mutable_split_parallel()->set_axis(0);
    }
  }
  return Maybe<void>::Ok();
}

Maybe<const LogicalBlobId*> JobBuildAndInferCtx::GetSubLbi(const LogicalBlobId& lbi,
                                                           int32_t index) {
  auto lbi_vec_iter = mirrored_lbi2sub_lbis_.find(lbi);
  if (lbi_vec_iter == mirrored_lbi2sub_lbis_.end()) {
    const auto& new_lbi = JUST(FindOrCreateMirroredLbiFromCompatibleConsistentBlob(lbi));
    lbi_vec_iter = mirrored_lbi2sub_lbis_.find(*new_lbi);
    CHECK(lbi_vec_iter != mirrored_lbi2sub_lbis_.end());
  }
  return &lbi_vec_iter->second.at(index);
}

// TODO(): add handle error of same interface op blob between jobs
Maybe<void> JobBuildAndInferCtx::AddAndInferConsistentOp(const OperatorConf& op_conf,
                                                         const ParallelConf& origin_parallel_conf) {
  CHECK_OR_RETURN(has_job_conf_) << JobBuildAndInferError::kJobConfNotSet;
  if (!is_job_conf_frozen_) { is_job_conf_frozen_ = true; }
  const std::string& op_name = op_conf.name();
  CHECK_OR_RETURN(op_name2op_.find(op_name) == op_name2op_.end())
      << JobBuildAndInferError::kOpNameExist << "op_name: " << op_name
      << " already exist in job: " << job_->job_conf().job_name();
  CHECK_NE_OR_RETURN(op_conf.device_type(), DeviceType::kInvalidDevice)
      << JobBuildAndInferError::kOpConfDeviceTypeNoSet << "op_name: " << op_name
      << " not set device type";

  op_name2op_.emplace(op_name, ConstructOp(op_conf, &GlobalJobDesc()));
  Operator* op = op_name2op_.at(op_name).get();

  SbpSignature sbp_sig_conf;
  HashMap<std::string, bool> ibn2disable_boxing;
  InitIbn2DisableBoxing(*op, &ibn2disable_boxing);
  auto new_op_conf = JUST(DecodeLbiHintAndReturnNewOpConf(*op, &sbp_sig_conf, &ibn2disable_boxing));
  AddOp7AddSbpSigConf2Job(*new_op_conf, sbp_sig_conf);
  auto parallel_conf = JUST(InferOpParallelConf(*op, origin_parallel_conf, ibn2disable_boxing));
  JUST(AddOpNameParallelConf2Placement(op_name, *parallel_conf));
  UpdateLbi2DisableBoxing(*op, ibn2disable_boxing);
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
  ParallelDesc parallel_desc(*parallel_conf);
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

bool JobBuildAndInferCtx::HasJobConf() const { return has_job_conf_; }

Maybe<void> JobBuildAndInferCtx::AddLossLogicalBlobName(const std::string& lbn) {
  if (IsMirroredBlob(lbn)) { return AddLossMirroredBlobName(lbn); }
  return AddLossConsistentBlobName(lbn);
}

Maybe<void> JobBuildAndInferCtx::AddLossConsistentBlobName(const std::string& lbn) {
  JUST(CheckLbnValidAndExist(lbn));
  CHECK_OR_RETURN(job_->job_conf().has_train_conf())
      << JobBuildAndInferError::kUnknownJobBuildAndInferError
      << "job has not TrainConf when add loss logical blob name";
  job_->mutable_job_conf()->mutable_train_conf()->add_loss_lbn(lbn);
  return Maybe<void>::Ok();
}

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

Maybe<bool> JobBuildAndInferCtx::IsTensorList(const std::string& lbn) const {
  JUST(CheckLbnValidAndExist(lbn));
  return lbi2logical_blob_desc_.at(GenLogicalBlobId(lbn))->is_tensor_list();
}

Maybe<bool> JobBuildAndInferCtx::DisableBoxing(const std::string& lbn) const {
  JUST(CheckLbnValidAndExist(lbn));
  LogicalBlobId lbi(GenLogicalBlobId(lbn));
  const auto& iter = lbi2disable_boxing_.find(lbi);
  OF_CHECK(iter != lbi2disable_boxing_.end());
  return iter->second;
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

Maybe<void> JobBuildAndInferCtx::AddLossMirroredBlobName(const std::string& lbn) {
  const auto& mirrored_lbi = JUST(GetMirroredLbi(lbn));
  CHECK_OR_RETURN(job_->job_conf().has_train_conf())
      << JobBuildAndInferError::kUnknownJobBuildAndInferError
      << "job has not TrainConf when add loss logical blob name";
  for (const auto& lbi : mirrored_lbi2sub_lbis_.at(*mirrored_lbi)) {
    job_->mutable_job_conf()->mutable_train_conf()->add_loss_lbn(GenLogicalBlobName(lbi));
  }
  return Maybe<void>::Ok();
}

Maybe<LogicalBlobId> JobBuildAndInferCtx::GetMirroredLbi(const std::string& lbn_with_hint) const {
  const LogicalBlobId& lbi = GenLogicalBlobId(lbn_with_hint);
  if (mirrored_lbi2sub_lbis_.find(lbi) != mirrored_lbi2sub_lbis_.end()) { return lbi; }
  return Error::CheckFailed() << lbn_with_hint << " is not a mirrored blob name";
}

Maybe<int> JobBuildAndInferCtx::MirroredBlobGetNumSubLbi(const std::string& lbn_with_hint) const {
  const auto& mirrored_lbi = JUST(GetMirroredLbi(lbn_with_hint));
  return mirrored_lbi2sub_lbis_.at(*mirrored_lbi).size();
}

Maybe<const LogicalBlobId*> JobBuildAndInferCtx::MirroredBlobGetSubLbi(
    const std::string& lbn_with_hint, int index) const {
  const auto& mirrored_lbi = JUST(GetMirroredLbi(lbn_with_hint));
  const auto& vec = mirrored_lbi2sub_lbis_.at(*mirrored_lbi);
  OF_CHECK_GE(index, 0);
  OF_CHECK_LT(index, vec.size());
  return &vec.at(index);
}

bool JobBuildAndInferCtx::IsMirroredBlob(const std::string& lbn) const {
  return GetMirroredLbi(lbn).IsOk();
}

Maybe<Shape> JobBuildAndInferCtx::MirroredBlobGetStaticShape(
    const std::string& lbn_with_hint) const {
  const auto& lbi = *JUST(MirroredBlobGetSubLbi(lbn_with_hint, 0));
  return lbi2logical_blob_desc_.at(lbi)->shape();
}

Maybe<DataType> JobBuildAndInferCtx::MirroredBlobGetDataType(
    const std::string& lbn_with_hint) const {
  const auto& lbi = *JUST(MirroredBlobGetSubLbi(lbn_with_hint, 0));
  return lbi2logical_blob_desc_.at(lbi)->data_type();
}

Maybe<bool> JobBuildAndInferCtx::MirroredBlobIsDynamic(const std::string& lbn_with_hint) const {
  const auto& lbi = *JUST(MirroredBlobGetSubLbi(lbn_with_hint, 0));
  return lbi2logical_blob_desc_.at(lbi)->is_dynamic();
}

Maybe<bool> JobBuildAndInferCtx::MirroredBlobIsTensorList(const std::string& lbn_with_hint) const {
  const auto& lbi = *JUST(MirroredBlobGetSubLbi(lbn_with_hint, 0));
  return lbi2logical_blob_desc_.at(lbi)->is_tensor_list();
}

Maybe<bool> JobBuildAndInferCtx::MirroredBlobDisableBoxing(const std::string& lbn_with_hint) const {
  OF_CHECK(IsMirroredBlob(lbn_with_hint));
  return true;
}

Maybe<OptInt64> JobBuildAndInferCtx::MirroredBlobGetBatchAxis(
    const std::string& lbn_with_hint) const {
  OF_CHECK(IsMirroredBlob(lbn_with_hint));
  auto ret = std::make_shared<OptInt64>();
  ret->set_value(0);
  return ret;
}

Maybe<OptInt64> JobBuildAndInferCtx::MirroredBlobGetSplitAxisFromProducerView(
    const std::string& lbn_with_hint) const {
  const auto& lbi = *JUST(MirroredBlobGetSubLbi(lbn_with_hint, 0));
  OptInt64 ret;
  const auto& sbp = lbi2sbp_parallel_from_producer_view_.at(lbi);
  if (sbp.has_split_parallel()) { ret.set_value(sbp.split_parallel().axis()); }
  return ret;
}

Maybe<const ParallelDesc*> JobBuildAndInferCtx::MirroredBlobGetParallelDescFromProducerView(
    const std::string& lbn_with_hint) const {
  const auto& lbi = JUST(GetMirroredLbi(lbn_with_hint));
  return &(mirrored_lbi2parallel_desc_.at(*lbi));
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
