/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/mirrored_sig_infer_hint.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/user/summary/summary_converter.h"

#include <google/protobuf/text_format.h>
#include <json.hpp>

namespace oneflow {

static const std::string kAutoMirroredBlobNamePrefix =
    "System-Mirrored-Blob-Auto-Converted-From-Consistent-Blob";

namespace {

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

Maybe<void> GetOpNames(const Job& job, HashSet<std::string>* op_names) {
  for (const auto& op_conf : job.net().op()) {
    CHECK_OR_RETURN(op_names->insert(op_conf.name()).second);
  }
  return Maybe<void>::Ok();
}

Maybe<void> EagerRunOps(const Job& job, HashSet<std::string>* op_names,
                        void (ForeignCallback::*interpret)(
                            const std::shared_ptr<cfg::OpAttribute>& op_attribute,
                            const std::shared_ptr<cfg::ParallelConf>& parallel_conf) const) {
  const auto& op_graph = JUST(OpGraph::New(job));
  const auto* foreign_callback = JUST(GlobalMaybe<ForeignCallback>());
  JUST(op_graph->ForEachOpNode([&](const OpNode& op_node) -> Maybe<void> {
    if (!op_names->insert(op_node.op().op_name()).second) { return Maybe<void>::Ok(); }
    const auto& op_attribute = op_node.op().GetOpAttributeWithoutOpNameAndLbn();
    const auto& parallel_conf = op_node.parallel_desc().parallel_conf();
    {
      const std::shared_ptr<cfg::OpAttribute>& cfg_op_attribute =
          std::make_shared<cfg::OpAttribute>(*op_attribute);
      const std::shared_ptr<cfg::ParallelConf>& cfg_parallel_conf =
          std::make_shared<cfg::ParallelConf>(parallel_conf);
      (foreign_callback->*interpret)(cfg_op_attribute, cfg_parallel_conf);
    }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

void UpdateOpName2AncestorsNeedNoGrad(
    const Operator& op, const std::function<const Operator*(const std::string&)>& Op4OpName,
    HashMap<std::string, bool>* op_name2ancestors_need_no_grad) {
  bool no_grad = op.job_desc().IsPredict();
  auto IsTrainableVariableLbi = [&](const LogicalBlobId& lbi) {
    const auto& op_conf = Op4OpName(lbi.op_name())->op_conf();
    return op_conf.has_variable_conf() && op_conf.trainable();
  };
  for (const auto& ibn : op.input_bns()) {
    const auto& lbi = op.BnInOp2Lbi(ibn);
    no_grad = no_grad && !IsTrainableVariableLbi(lbi);
    no_grad = no_grad && !op.InputBlobModifier4Ibn(ibn).requires_grad();
    no_grad = no_grad && (*op_name2ancestors_need_no_grad)[lbi.op_name()];
  }
  (*op_name2ancestors_need_no_grad)[op.op_name()] = no_grad;
}

}  // namespace

JobBuildAndInferCtx::JobBuildAndInferCtx(Job* job, int64_t job_id) : job_(job), job_id_(job_id) {
  is_job_conf_frozen_ = false;
  has_job_conf_ = false;
}

Maybe<void> JobBuildAndInferCtx::SetJobConf(const JobConfigProto& job_conf) {
  CHECK_OR_RETURN(!is_job_conf_frozen_) << Error::JobConfFrozenError();
  CHECK_OR_RETURN(!has_job_conf_) << Error::JobConfRepeatedSetError();
  has_job_conf_ = true;
  CHECK_EQ_OR_RETURN(job_->job_conf().job_name(), job_conf.job_name())
      << Error::JobNameNotEqualError() << "job name you set: " << job_conf.job_name()
      << " not equal to origin job name: " << job_->job_conf().job_name();
  job_->mutable_job_conf()->CopyFrom(job_conf);
  CHECK_ISNULL_OR_RETURN(Global<JobDesc>::Get());
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
    CHECK_OR_RETURN(std::find(iter->second->lbi().begin(), iter->second->lbi().end(), lbi)
                    == iter->second->lbi().end());
    *iter->second->add_lbi() = lbi;
  }
  return Maybe<void>::Ok();
}

Maybe<OperatorConf> JobBuildAndInferCtx::DecodeLbiHintAndReturnNewOpConf(
    const Operator& op, SbpSignature* sbp_sig_conf,
    HashMap<std::string, bool>* ibn2disable_boxing) const {
  auto op_conf_without_split_hint = std::make_shared<OperatorConf>(op.op_conf());
  for (const std::string& ibn : op.input_bns()) {
    std::string lbn_may_with_hint = GetInputLbnInOpCustomizedConf(op.op_conf(), ibn);
    SbpParallel sbp_parallel;
    bool has_sbp_hint = JUST(GetSbpParallelInLbnOrNothing(lbn_may_with_hint, &sbp_parallel));
    bool has_disable_boxing_hint =
        JUST(ParseDisableBoxingFlag(lbn_may_with_hint, &(*ibn2disable_boxing)[ibn]));
    if (has_sbp_hint || has_disable_boxing_hint) {
      (*(sbp_sig_conf->mutable_bn_in_op2sbp_parallel()))[ibn] = sbp_parallel;
      const LogicalBlobId& lbi = op.BnInOp2Lbi(ibn);
      std::string lbn = GenLogicalBlobName(lbi);
      CHECK_EQ_OR_RETURN(lbn_may_with_hint, ReplaceInputLbnInOpCustomizedConf(
                                                op_conf_without_split_hint.get(), ibn, lbn));
    }
  }
  return op_conf_without_split_hint;
}

void JobBuildAndInferCtx::AddOpAndUpdateJobParallelViewConf(const OperatorConf& operator_conf,
                                                            const SbpSignature& sbp_signature,
                                                            bool is_mirrored_parallel_view) const {
  auto* op_name2sbp_sig =
      job_->mutable_job_parallel_view_conf()->mutable_op_name2sbp_signature_conf();
  if (sbp_signature.bn_in_op2sbp_parallel().size() > 0) {
    (*op_name2sbp_sig)[operator_conf.name()] = sbp_signature;
  }
  auto* op_name2is_mirrored_parallel_view =
      job_->mutable_job_parallel_view_conf()->mutable_op_name2is_mirrored_parallel_view();
  if (is_mirrored_parallel_view) {
    (*op_name2is_mirrored_parallel_view)[operator_conf.name()] = true;
  }
  job_->mutable_net()->add_op()->CopyFrom(operator_conf);
}

Maybe<void> JobBuildAndInferCtx::InferMirroredSignature(Operator* op,
                                                        bool is_mirrored_parallel_view_conf,
                                                        const ParallelDesc& parallel_desc) {
  HashMap<std::string, MirroredSigInferHint> ibn2mirrored_sig_infer_hint;
  for (const std::string& ibn : op->input_bns()) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(ibn);
    CHECK_OR_RETURN(lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end())
        << Error::LogicalBlobNameNotExistError()
        << "infer blob desc not found, when infer op_name: \"" << op->op_name()
        << "\", consumed op_name: \"" << lbi.op_name() << "\", blob_name: \"" << lbi.blob_name();
    const ParallelDesc* pd = &lbi2parallel_desc_from_producer_view_.at(lbi);
    const auto* producer_op = op_name2op_.at(lbi.op_name()).get();
    const auto& producer_obn = *JUST(producer_op->obn4lbi(lbi));
    const auto& opt_mirrored_parallel =
        *CHECK_JUST(producer_op->OptMirroredParallel4BnInOp(producer_obn));
    ibn2mirrored_sig_infer_hint.emplace(
        ibn, MirroredSigInferHint(pd, opt_mirrored_parallel.has_mirrored_parallel()));
  }
  const auto& MirroredSigInferHint4Ibn =
      [&](const std::string& ibn) -> Maybe<const MirroredSigInferHint*> {
    const auto& iter = ibn2mirrored_sig_infer_hint.find(ibn);
    CHECK_OR_RETURN(iter != ibn2mirrored_sig_infer_hint.end())
        << "input blob not found. ibn: " << ibn;
    return &iter->second;
  };
  JUST(op->InferMirroredSignatureIf(MirroredSigInferHint4Ibn, is_mirrored_parallel_view_conf,
                                    parallel_desc));
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::InferOpOutSbpParallel(Operator* op,
                                                       const SbpSignature& sbp_sig_conf,
                                                       const ParallelDesc& parallel_desc) {
  HashMap<std::string, SbpInferHint> ibn2sbp_infer_hint;
  for (const std::string& ibn : op->input_bns()) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(ibn);
    CHECK_OR_RETURN(lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end())
        << Error::LogicalBlobNameNotExistError()
        << "infer blob desc not found, when infer op_name: \"" << op->op_name()
        << "\", consumed op_name: \"" << lbi.op_name() << "\", blob_name: \"" << lbi.blob_name();
    const ParallelDesc* pd = &lbi2parallel_desc_from_producer_view_.at(lbi);
    const BlobDesc* logical_blob_desc = lbi2logical_blob_desc_.at(lbi).get();
    CHECK_OR_RETURN(lbi2sbp_parallel_from_producer_view_.find(lbi)
                    != lbi2sbp_parallel_from_producer_view_.end())
        << Error::LogicalBlobNameNotExistError() << "when infer op_name: " << op->op_name()
        << " consumed op_name: " << lbi.op_name() << " blob_name: " << lbi.blob_name()
        << " not infer split axis";
    const SbpParallel* sbp_parallel = &lbi2sbp_parallel_from_producer_view_.at(lbi);
    ibn2sbp_infer_hint.emplace(ibn, SbpInferHint(pd, logical_blob_desc, sbp_parallel));
  }

  JUST(InferOpSbpSignature(op, sbp_sig_conf, parallel_desc, ibn2sbp_infer_hint));

  const auto& bn2sbp_parallel = JUST(op->sbp_signature())->bn_in_op2sbp_parallel();
  for (const auto& obn : op->output_bns()) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(obn);
    CHECK_OR_RETURN(bn2sbp_parallel.find(obn) != bn2sbp_parallel.end())
        << Error::BlobSplitAxisInferError() << "op_name: " << lbi.op_name()
        << " blob_name: " << lbi.blob_name() << " not infer split axis";
    CHECK_OR_RETURN(
        lbi2sbp_parallel_from_producer_view_.emplace(lbi, bn2sbp_parallel.at(obn)).second)
        << Error::BlobSplitAxisInferError() << "op_name: " << lbi.op_name()
        << " blob_name: " << lbi.blob_name() << " infer split axis repeated";
    CHECK_OR_RETURN(lbi2parallel_desc_from_producer_view_.emplace(lbi, parallel_desc).second)
        << Error::BlobSplitAxisInferError() << "op_name: " << lbi.op_name()
        << " blob_name: " << lbi.blob_name() << " add parallel desc repeated";
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::GenOpProducedEmptyLogicalBlobDesc(Operator* op) {
  // check consumed blob
  for (const std::string& consumed_bn : op->input_bns()) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(consumed_bn);
    CHECK_OR_RETURN(lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end())
        << Error::LogicalBlobNameNotExistError() << "op_name: " << op->op_name()
        << " consumed_op_name:" << lbi.op_name() << " blob_name: " << lbi.blob_name()
        << " not exist";
  }

  // create produced blob
  std::vector<std::string> produced_bns;
  produced_bns.insert(produced_bns.end(), op->output_bns().begin(), op->output_bns().end());
  produced_bns.insert(produced_bns.end(), op->tmp_bns().begin(), op->tmp_bns().end());
  for (const std::string& produced_bn : produced_bns) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(produced_bn);
    CHECK_OR_RETURN(lbi2logical_blob_desc_.find(lbi) == lbi2logical_blob_desc_.end())
        << Error::LogicalBlobNameExistError()
        << "duplicate logical blob name found. op_name: " << lbi.op_name()
        << " blob_name: " << lbi.blob_name();
    lbi2logical_blob_desc_.emplace(lbi, std::make_unique<BlobDesc>(DataType::kInvalidDataType));
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::CheckOpBlobSplitability(Operator* op, int64_t parallel_num) {
  HashSet<std::string> obns(op->output_bns().begin(), op->output_bns().end());
  auto GetParallelNum = [&](const std::string& bn_in_op) {
    if (obns.find(bn_in_op) == obns.end()) { return parallel_num; }
    return lbi2parallel_desc_from_producer_view_.at(op->BnInOp2Lbi(bn_in_op)).parallel_num();
  };
  for (const auto& pair : JUST(op->sbp_signature())->bn_in_op2sbp_parallel()) {
    if (!pair.second.has_split_parallel()) { continue; }
    if (JUST(op->OptMirroredParallel4BnInOp(pair.first))->has_mirrored_parallel()) { continue; }
    int64_t axis = pair.second.split_parallel().axis();
    const LogicalBlobId& lbi = op->BnInOp2Lbi(pair.first);
    int64_t blob_parallel_num = GetParallelNum(pair.first);
    const BlobDesc& logical_blob_desc = *(lbi2logical_blob_desc_.at(lbi).get());
    int64_t num_axes = logical_blob_desc.shape().NumAxes();
    if (axis < 0) { axis += num_axes; }
    CHECK_GE_OR_RETURN(axis, 0);
    CHECK_LT_OR_RETURN(axis, num_axes)
        << "op: " << op->op_name() << ", blob: " << pair.first << ", axis: " << axis
        << ", shape: " << logical_blob_desc.shape();
    CHECK_GE_OR_RETURN(logical_blob_desc.shape().At(axis), blob_parallel_num)
        << "op_name: " << lbi.op_name() << " blob_name: " << lbi.blob_name()
        << " cannot split blob by parallel_num: " << std::to_string(blob_parallel_num);
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
      CHECK_EQ_OR_RETURN(parallel_desc->parallel_num(), ibn_parallel_desc.parallel_num());
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
  CHECK_OR_RETURN(iter != lbi2sbp_parallel_from_producer_view_.end())
      << "lbn: " << GenLogicalBlobName(lbi) << " undefined";
  return &iter->second;
}

Maybe<const ParallelDesc*> JobBuildAndInferCtx::ParallelDesc4Lbi(const LogicalBlobId& lbi) const {
  const auto& iter = lbi2parallel_desc_from_producer_view_.find(lbi);
  CHECK_OR_RETURN(iter != lbi2parallel_desc_from_producer_view_.end())
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

bool JobBuildAndInferCtx::IsVariableLbi(const LogicalBlobId& lbi) const {
  return op_name2op_.at(lbi.op_name())->op_conf().has_variable_conf();
}

Maybe<void> JobBuildAndInferCtx::CheckAllInputsConvertableToMirroredBlob(const Operator& op) const {
  for (const auto& ibn : op.input_bns()) {
    const auto& lbi = op.BnInOp2Lbi(ibn);
    if (mirrored_lbi2sub_lbis_.find(lbi) != mirrored_lbi2sub_lbis_.end()) { continue; }
    const auto& sbp = *JUST(SbpParallel4Lbi(lbi));
    if (sbp.has_broadcast_parallel()) { continue; }
    if (sbp.has_split_parallel() && sbp.split_parallel().axis() == 0) { continue; }
    const std::string& lbn = GenLogicalBlobName(lbi);
    return Error::CheckFailedError()
           << "input lbn: " << lbn << " is not convertable to mirrored blob";
  }
  return Maybe<void>::Ok();
}

Maybe<void> LazyJobBuildAndInferCtx::CheckAllInputsWithSameParallelNum(const Operator& op,
                                                                       int32_t parallel_num) const {
  for (const auto& ibn : op.input_bns()) {
    const auto& lbi = op.BnInOp2Lbi(ibn);
    const auto& iter = mirrored_lbi2sub_lbis().find(lbi);
    int32_t ibn_parallel_num = 0;
    if (iter != mirrored_lbi2sub_lbis().end()) {
      ibn_parallel_num = iter->second.size();
    } else {
      ibn_parallel_num = JUST(ParallelDesc4Lbi(lbi))->parallel_num();
    }
    CHECK_EQ_OR_RETURN(ibn_parallel_num, parallel_num)
        << "the parallel_num of input lbn: " << GenLogicalBlobName(lbi)
        << " is not equals to op' parallel_num";
  }
  return Maybe<void>::Ok();
}

Maybe<void> EagerJobBuildAndInferCtx::CheckAllInputsWithSameParallelNum(
    const Operator& op, int32_t parallel_num) const {
  for (const auto& ibn : op.input_bns()) {
    const auto& lbi = op.BnInOp2Lbi(ibn);
    int32_t ibn_parallel_num = JUST(ParallelDesc4Lbi(lbi))->parallel_num();
    CHECK_EQ_OR_RETURN(ibn_parallel_num, parallel_num)
        << "the parallel_num of input lbn: " << GenLogicalBlobName(lbi)
        << "is not equals to op' parallel_num";
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::AddLbiAndDiffWatcherUuidPair(
    const LbiAndDiffWatcherUuidPair& lbi_uuid_pair) {
  const auto& job_name = job_->job_conf().job_name();
  auto* job_helper = job_->mutable_helper();
  auto* job_name2pairs =
      job_helper->mutable_lbi_diff_watcher_info()->mutable_job_name2lbi_and_watcher_uuids();
  LbiAndDiffWatcherUuidPairList* pairs = &(*job_name2pairs)[job_name];
  auto PairFoundCond = [&](const LbiAndDiffWatcherUuidPair& x) {
    return x.lbi() == lbi_uuid_pair.lbi() && x.watcher_uuid() == lbi_uuid_pair.watcher_uuid();
  };
  auto found_iter = std::find_if(pairs->lbi_and_uuid_pair().begin(),
                                 pairs->lbi_and_uuid_pair().end(), PairFoundCond);
  CHECK_OR_RETURN(found_iter == pairs->lbi_and_uuid_pair().end())
      << "diff blob has been watched. (logical_blob_name: "
      << GenLogicalBlobName(lbi_uuid_pair.lbi()) << ", job_name: " << job_name << ")";
  *pairs->mutable_lbi_and_uuid_pair()->Add() = lbi_uuid_pair;
  return Maybe<void>::Ok();
}

Maybe<OpAttribute> JobBuildAndInferCtx::AddAndInferMirroredOp(const OperatorConf& op_conf) {
  CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
  const auto& scope = Global<symbol::Storage<Scope>>::Get()->Get(op_conf.scope_symbol_id());
  const auto* job_desc = JUST(scope.job_desc());
  const auto& parallel_desc = JUST(scope.GetParallelDesc(op_conf));
  auto op = ConstructOp(op_conf, parallel_desc.device_type(), job_desc);
  JUST(CheckAllInputsConvertableToMirroredBlob(*op));
  int32_t parallel_num = parallel_desc.parallel_num();
  JUST(CheckAllInputsWithSameParallelNum(*op, parallel_num));
  auto GetSubOpName = [&](int index) { return GetMirroredOpName(op_conf.name(), index); };
  OperatorConf sub_op_conf(op_conf);
  int64_t sub_op_list_size = SizeOfSubConsistentOpList(parallel_num);
  auto last_op_attribute = std::make_shared<OpAttribute>();
  FOR_RANGE(int32_t, i, 0, sub_op_list_size) {
    ResetOpConfName(&sub_op_conf, GetSubOpName(i));
    for (const auto& ibn : op->input_bns()) {
      const auto& lbi = *JUST(GetSubLbi(op_conf.scope_symbol_id(), op->BnInOp2Lbi(ibn), i));
      ReplaceInputLbnInOpCustomizedConf(&sub_op_conf, ibn, GenLogicalBlobName(lbi));
    }
    const ParallelConf& parallel_conf = GetMirroredOpParallelConf(parallel_desc, i);
    bool is_mirrored_parallel_view = GetIsMirroredParallelView();
    last_op_attribute =
        JUST(AddAndInferOp(sub_op_conf, parallel_conf, job_desc, is_mirrored_parallel_view));
  }
  bool is_broadcast = JUST(AllInputsBroadcastParallel(*op));
  for (const auto& obn : op->output_bns()) {
    const auto& lbi = op->BnInOp2Lbi(obn);
    auto* sub_lbis = &mirrored_lbi2sub_lbis_[lbi];
    sub_lbis->resize(sub_op_list_size, op->BnInOp2Lbi(obn));
    FOR_RANGE(int32_t, i, 0, sub_op_list_size) { sub_lbis->at(i).set_op_name(GetSubOpName(i)); }
    CHECK(mirrored_lbi2parallel_desc_.emplace(lbi, parallel_desc).second);
    auto* sbp_parallel = &mirrored_lbi2sbp_parallel_[lbi];
    if (is_broadcast) {
      sbp_parallel->mutable_broadcast_parallel();
    } else {
      sbp_parallel->mutable_split_parallel()->set_axis(0);
    }
  }
  return last_op_attribute;
}

Maybe<const LogicalBlobId*> JobBuildAndInferCtx::GetSubLbi(int64_t scope_symbol_id,
                                                           const LogicalBlobId& lbi,
                                                           int32_t index) {
  auto lbi_vec_iter = mirrored_lbi2sub_lbis_.find(lbi);
  if (lbi_vec_iter == mirrored_lbi2sub_lbis_.end()) {
    const auto& new_lbi =
        JUST(FindOrCreateMirroredLbiFromCompatibleConsistentBlob(scope_symbol_id, lbi));
    lbi_vec_iter = mirrored_lbi2sub_lbis_.find(*new_lbi);
    CHECK(lbi_vec_iter != mirrored_lbi2sub_lbis_.end());
  }
  return &lbi_vec_iter->second.at(index);
}

Maybe<OpAttribute> JobBuildAndInferCtx::AddAndInferConsistentOp(const OperatorConf& op_conf) {
  CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
  const auto& scope = Global<symbol::Storage<Scope>>::Get()->Get(op_conf.scope_symbol_id());
  const auto& parallel_desc = JUST(scope.GetParallelDesc(op_conf));
  const auto* job_desc = JUST(scope.job_desc());
  return AddAndInferOp(op_conf, parallel_desc.parallel_conf(), job_desc, false);
}

// TODO(): add handle error of same interface op blob between jobs
Maybe<OpAttribute> JobBuildAndInferCtx::AddAndInferOp(const OperatorConf& op_conf,
                                                      const ParallelConf& origin_parallel_conf,
                                                      const JobDesc* job_desc,
                                                      bool is_mirrored_parallel_view) {
  CHECK_OR_RETURN(has_job_conf_) << Error::JobConfNotSetError();
  if (!is_job_conf_frozen_) { is_job_conf_frozen_ = true; }
  const std::string& op_name = op_conf.name();
  CHECK_OR_RETURN(op_name2op_.find(op_name) == op_name2op_.end())
      << Error::OpNameExistError() << "op_name: " << op_name
      << " already exist in job: " << job_->job_conf().job_name();
  CHECK_NE_OR_RETURN(op_conf.device_tag(), "invalid_device")
      << Error::OpConfDeviceTagNoSetError() << "op_name: " << op_name << " not set device tag";

  op_name2op_.emplace(op_name, ConstructOp(op_conf, job_desc));
  Operator* op = op_name2op_.at(op_name).get();

  SbpSignature sbp_sig_conf;
  HashMap<std::string, bool> ibn2disable_boxing;
  InitIbn2DisableBoxing(*op, &ibn2disable_boxing);
  auto new_op_conf = JUST(DecodeLbiHintAndReturnNewOpConf(*op, &sbp_sig_conf, &ibn2disable_boxing));
  AddOpAndUpdateJobParallelViewConf(*new_op_conf, sbp_sig_conf, is_mirrored_parallel_view);
  auto parallel_conf = JUST(InferOpParallelConf(*op, origin_parallel_conf, ibn2disable_boxing));
  ParallelDesc parallel_desc(*parallel_conf);
  JUST(op->FillOpParallelDesc(parallel_desc));
  JUST(AddOpNameParallelConf2Placement(op_name, *parallel_conf));
  UpdateLbi2DisableBoxing(*op, ibn2disable_boxing);

  auto GetBlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(bn);
    if (lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end()) {
      return lbi2logical_blob_desc_.at(lbi).get();
    }
    return nullptr;
  };
  JUST(op->FillLogicalInBlobDesc(GetBlobDesc4BnInOp));

  // infer mirrored signature
  JUST(InferMirroredSignature(op, is_mirrored_parallel_view, parallel_desc));
  // infer sbp signature
  JUST(InferOpOutSbpParallel(op, sbp_sig_conf, parallel_desc));

  // infer logical blob desc
  JUST(GenOpProducedEmptyLogicalBlobDesc(op));
  JUST(op->InferLogicalOutBlobDescsIf(GetBlobDesc4BnInOp, parallel_desc));
  JUST(op->FillLogicalOutBlobDesc(GetBlobDesc4BnInOp));
  // Infer ParallelDesc for output blobs.
  auto ParallelDesc4Obn = [&](const std::string& obn) -> ParallelDesc* {
    const auto& lbi = op->BnInOp2Lbi(obn);
    auto iter = lbi2parallel_desc_from_producer_view_.find(lbi);
    if (iter == lbi2parallel_desc_from_producer_view_.end()) {
      iter = lbi2parallel_desc_from_producer_view_.emplace(lbi, parallel_desc).first;
    }
    return &iter->second;
  };
  JUST(op->InferParallelSignatureIf());
  for (const auto& bn : op->output_bns()) {
    lbi2parallel_desc_from_producer_view_.emplace(op->BnInOp2Lbi(bn),
                                                  *JUST(op->GetParallelDesc4BnInOp(bn)));
  }
  JUST(AddLbiParallelConf2BlobPlacement(op, ParallelDesc4Obn));
  // Infer whether input/output blobs are backward used
  InferBlobBackwardSignature(op);
  // Check splitability
  JUST(CheckOpBlobSplitability(op, parallel_desc.parallel_num()));

  return op->GetOpAttributeWithoutOpNameAndLbn();
}

bool JobBuildAndInferCtx::HasJobConf() const { return has_job_conf_; }

Maybe<void> JobBuildAndInferCtx::SetTrainConf(const TrainConf& train_conf) {
  *job_->mutable_job_conf()->mutable_train_conf() = train_conf;
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::AddLossLogicalBlobName(const std::string& lbn) {
  if (IsMirroredBlob(lbn)) { return AddLossMirroredBlobName(lbn); }
  return AddLossConsistentBlobName(lbn);
}

Maybe<void> JobBuildAndInferCtx::AddLossConsistentBlobName(const std::string& lbn) {
  JUST(CheckLbnValidAndExist(lbn));
  CHECK_OR_RETURN(job_->job_conf().has_train_conf())
      << Error::UnknownJobBuildAndInferError()
      << "job has no TrainConf when adding loss logical blob name";
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
  CHECK_OR_RETURN(iter != lbi2disable_boxing_.end());
  return iter->second;
}

Maybe<Operator*> JobBuildAndInferCtx::Op4OpName(const std::string& op_name) const {
  const auto& op_iter = op_name2op_.find(op_name);
  CHECK_OR_RETURN(op_iter != op_name2op_.end());
  auto* op = op_iter->second.get();
  CHECK_NOTNULL_OR_RETURN(op);
  return op;
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
      << Error::UnknownJobBuildAndInferError()
      << "job has no TrainConf when adding loss logical blob name";
  for (const auto& lbi : mirrored_lbi2sub_lbis_.at(*mirrored_lbi)) {
    job_->mutable_job_conf()->mutable_train_conf()->add_loss_lbn(GenLogicalBlobName(lbi));
  }
  return Maybe<void>::Ok();
}

Maybe<LogicalBlobId> JobBuildAndInferCtx::GetMirroredLbi(const std::string& lbn_with_hint) const {
  const LogicalBlobId& lbi = GenLogicalBlobId(lbn_with_hint);
  if (mirrored_lbi2sub_lbis_.find(lbi) != mirrored_lbi2sub_lbis_.end()) { return lbi; }
  return Error::CheckFailedError() << lbn_with_hint << " is not a mirrored blob name";
}

Maybe<int> JobBuildAndInferCtx::MirroredBlobGetNumSubLbi(const std::string& lbn_with_hint) const {
  const auto& mirrored_lbi = JUST(GetMirroredLbi(lbn_with_hint));
  return mirrored_lbi2sub_lbis_.at(*mirrored_lbi).size();
}

Maybe<const LogicalBlobId*> JobBuildAndInferCtx::MirroredBlobGetSubLbi(
    const std::string& lbn_with_hint, int index) const {
  const auto& mirrored_lbi = JUST(GetMirroredLbi(lbn_with_hint));
  const auto& vec = mirrored_lbi2sub_lbis_.at(*mirrored_lbi);
  CHECK_GE_OR_RETURN(index, 0);
  CHECK_LT_OR_RETURN(index, vec.size());
  return &vec.at(index);
}

bool JobBuildAndInferCtx::IsMirroredBlob(const std::string& lbn) const {
  bool is_mirrored_blob = TRY(GetMirroredLbi(lbn)).IsOk();
  if (is_mirrored_blob) { return is_mirrored_blob; }
  const LogicalBlobId& lbi = GenLogicalBlobId(lbn);
  CHECK(lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end()) << "lbn: " << lbn;
  return false;
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
        << Error::OpNameExistError() << "op_name: " << op_conf.name()
        << " already exist in job: " << job_->job_conf().job_name() << " net";
  }
  for (const PlacementGroup& placement_group : job_->placement().placement_group()) {
    for (const std::string& op_name : placement_group.op_set().op_name()) {
      CHECK_OR_RETURN(op_names_in_placement.insert(op_name).second)
          << Error::OpNameExistError() << "op_name: " << op_name
          << " already exist in job: " << job_->job_conf().job_name() << " placement";
    }
  }
  CHECK_EQ_OR_RETURN(op_names_in_net.size(), op_names_in_placement.size())
      << Error::PlacementError() << "job: " << job_->job_conf().job_name()
      << " op number not equal between net and placement";
  for (const std::string& op_name : op_names_in_net) {
    CHECK_OR_RETURN(op_names_in_placement.find(op_name) != op_names_in_placement.end())
        << Error::PlacementError() << "job: " << job_->job_conf().job_name()
        << " op_name: " << op_name << " defined in net cannot find its placement";
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::CheckJobConf() const {
  if (job_->job_conf().job_type_case() == JobConfigProto::JOB_TYPE_NOT_SET) {
    return Error::JobTypeNotSetError() << "job_type not set, please set predict_conf or train_conf";
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::CheckLbnValidAndExist(const std::string& lbn) const {
  CHECK_OR_RETURN(lbn.find('/') != std::string::npos)
      << Error::LogicalBlobNameInvalidError() << "lbn:" << lbn;
  LogicalBlobId lbi = GenLogicalBlobId(lbn);

#define CHECK_HAS_LBI_KEY(info_src)                     \
  CHECK_OR_RETURN(info_src.find(lbi) != info_src.end()) \
      << Error::LogicalBlobNameNotExistError() << "lbn:" << lbn;

  CHECK_HAS_LBI_KEY(lbi2logical_blob_desc_);
  CHECK_HAS_LBI_KEY(lbi2sbp_parallel_from_producer_view_);
  CHECK_HAS_LBI_KEY(lbi2parallel_desc_from_producer_view_);
#undef CHECK_HAS_LBI_KEY

  return Maybe<void>::Ok();
}

const Job& JobBuildAndInferCtx::job() const { return *job_; }

std::string LazyJobBuildAndInferCtx::GetMirroredOpName(const std::string& op_name,
                                                       int64_t parallel_id) const {
  return op_name + "_" + std::to_string(parallel_id);
}

std::string EagerJobBuildAndInferCtx::GetMirroredOpName(const std::string& op_name,
                                                        int64_t parallel_id) const {
  return op_name;
}

ParallelConf LazyJobBuildAndInferCtx::GetMirroredOpParallelConf(const ParallelDesc& parallel_desc,
                                                                int64_t parallel_id) const {
  return parallel_desc.GetParallelIdOnlyParallelConf(parallel_id);
}

ParallelConf EagerJobBuildAndInferCtx::GetMirroredOpParallelConf(const ParallelDesc& parallel_desc,
                                                                 int64_t parallel_id) const {
  return parallel_desc.parallel_conf();
}

Maybe<LogicalBlobId> LazyJobBuildAndInferCtx::FindOrCreateMirroredLbiFromCompatibleConsistentBlob(
    int64_t scope_symbol_id, const LogicalBlobId& lbi) {
  const std::string& lbn = GenLogicalBlobName(lbi);
  const auto& sbn_it = mut_consistent_lbi2mirrored_lbi()->find(lbi);
  if (sbn_it != mut_consistent_lbi2mirrored_lbi()->end()) { return sbn_it->second; }
  const SbpParallel& sbp = *JUST(SbpParallel4Lbi(lbi));
  const ParallelDesc& parallel_desc = *JUST(ParallelDesc4Lbi(lbi));
  LogicalBlobId mirrored_lbi;
  mirrored_lbi.set_op_name(kAutoMirroredBlobNamePrefix + NewUniqueId());
  mirrored_lbi.set_blob_name("out");
  (*mut_consistent_lbi2mirrored_lbi())[lbi] = mirrored_lbi;
  auto* lbi_vec = &(*mut_mirrored_lbi2sub_lbis())[mirrored_lbi];
  lbi_vec->reserve(parallel_desc.parallel_num());
  auto PushBackSubLbi = [&](const std::string& op_name, const std::string& blob_name) {
    LogicalBlobId sub_lbi;
    sub_lbi.set_op_name(op_name);
    sub_lbi.set_blob_name(blob_name);
    lbi_vec->push_back(sub_lbi);
  };
  OperatorConf op_conf;
  op_conf.set_scope_symbol_id(scope_symbol_id);
  op_conf.set_device_tag(CHECK_JUST(DeviceTag4DeviceType(parallel_desc.device_type())));
  if (sbp.has_broadcast_parallel()) {
    op_conf.set_name(kAutoMirroredBlobNamePrefix + "-DistributeClone-" + NewUniqueId());
    auto* distribute_clone = op_conf.mutable_distribute_clone_conf();
    distribute_clone->set_in(lbn);
    FOR_RANGE(int32_t, i, 0, parallel_desc.parallel_num()) {
      const std::string& blob_name = "out_" + std::to_string(i);
      distribute_clone->add_out(blob_name);
      distribute_clone->set_is_variable_ref(IsVariableLbi(lbi));
      PushBackSubLbi(op_conf.name(), blob_name);
    }
  } else if (sbp.has_split_parallel()) {
    CHECK_EQ_OR_RETURN(sbp.split_parallel().axis(), 0)
        << "only `S(0)' consistent blob is compatible to mirrored blob";
    op_conf.set_name(kAutoMirroredBlobNamePrefix + "-DistributeSplit-" + NewUniqueId());
    auto* distribute_split = op_conf.mutable_distribute_split_conf();
    distribute_split->set_in(lbn);
    distribute_split->set_axis(0);
    distribute_split->set_is_variable_ref(IsVariableLbi(lbi));
    FOR_RANGE(int32_t, i, 0, parallel_desc.parallel_num()) {
      const std::string& blob_name = "out_" + std::to_string(i);
      distribute_split->add_out(blob_name);
      PushBackSubLbi(op_conf.name(), blob_name);
    }
  } else {
    OF_UNIMPLEMENTED() << "`P' consistant blob is not compatible to mirrored blob";
  }
  {
    const auto& producer_op_conf = JUST(Op4OpName(lbi.op_name()))->op_conf();
    CHECK_OR_RETURN(producer_op_conf.has_scope_symbol_id());
    const auto& scope = Global<symbol::Storage<Scope>>::Get()->Get(scope_symbol_id);
    const auto* job_desc = JUST(scope.job_desc());
    JUST(AddAndInferOp(op_conf, parallel_desc.parallel_conf(), job_desc, false));
  }
  return mirrored_lbi;
}

Maybe<LogicalBlobId> EagerJobBuildAndInferCtx::FindOrCreateMirroredLbiFromCompatibleConsistentBlob(
    int64_t scope_symbol_id, const LogicalBlobId& lbi) {
  const std::string& lbn = GenLogicalBlobName(lbi);
  const auto& sbn_it = mut_consistent_lbi2mirrored_lbi()->find(lbi);
  if (sbn_it != mut_consistent_lbi2mirrored_lbi()->end()) { return sbn_it->second; }
  const SbpParallel& sbp = *JUST(SbpParallel4Lbi(lbi));
  CHECK_OR_RETURN(!sbp.has_partial_sum_parallel())
      << "`P' consistant blob is not compatible to mirrored blob";
  const ParallelDesc& parallel_desc = *JUST(ParallelDesc4Lbi(lbi));
  OperatorConf op_conf;
  {
    // inherit scope_symbol_id from producer
    const auto& producer_op_conf = JUST(Op4OpName(lbi.op_name()))->op_conf();
    CHECK_OR_RETURN(producer_op_conf.has_scope_symbol_id());
    op_conf.set_scope_symbol_id(producer_op_conf.scope_symbol_id());
  }
  op_conf.set_scope_symbol_id(scope_symbol_id);
  // const char* device_tag = JUST(DeviceTag4DeviceType(parallel_desc.device_type()));
  op_conf.set_device_tag(JUST(DeviceTag4DeviceType(parallel_desc.device_type())));
  op_conf.set_name(kAutoMirroredBlobNamePrefix + "-CastToMirrored-" + NewUniqueId());
  auto* cast_to_mirrored_conf = op_conf.mutable_cast_to_mirrored_conf();
  cast_to_mirrored_conf->set_in(lbn);
  cast_to_mirrored_conf->set_out("out");
  *cast_to_mirrored_conf->mutable_sbp_parallel() = sbp;
  LogicalBlobId mirrored_lbi;
  mirrored_lbi.set_op_name(op_conf.name());
  mirrored_lbi.set_blob_name("out");
  (*mut_consistent_lbi2mirrored_lbi())[lbi] = mirrored_lbi;
  (*mut_mirrored_lbi2sub_lbis())[mirrored_lbi].push_back(mirrored_lbi);
  const auto& parallel_conf = parallel_desc.parallel_conf();
  const auto& op_attribute = JUST(AddAndInferConsistentOp(op_conf));
  {
    const std::shared_ptr<cfg::OpAttribute>& cfg_op_attribute =
        std::make_shared<cfg::OpAttribute>(*op_attribute);
    const std::shared_ptr<cfg::ParallelConf>& cfg_parallel_conf =
        std::make_shared<cfg::ParallelConf>(parallel_conf);
    JUST(GlobalMaybe<ForeignCallback>())->EagerMirroredCast(cfg_op_attribute, cfg_parallel_conf);
  }
  return mirrored_lbi;
}

Maybe<void> LazyJobBuildAndInferCtx::Complete() {
  CHECK_NOTNULL(Global<JobDesc>::Get());
  Global<JobDesc>::Delete();
  auto scope = std::make_unique<GlobalJobDescScope>(mut_job()->job_conf(), job_id());
  JobPassCtx job_pass_ctx(GlobalJobDesc());
  auto DoPass = [&](const std::string& pass_name) -> Maybe<void> {
    return JobPass4Name(pass_name)(mut_job(), &job_pass_ctx);
  };
  if (GlobalJobDesc().Bool("__is_user_function__")) {
    JUST(DoPass("SetDefaultVariableConf"));
    JUST(DoPass("AddInputOutputOpsPass"));
#ifdef WITH_CUDA
    JUST(DoPass("AutoMixedPrecision"));
#endif
    JUST(DoPass("OptimizerPlacementOptimizationPass"));
    JUST(DoPass("DynamicLossScaleSchedulePass"));
    JUST(DoPass("AutoTrainStep"));
    JUST(DoPass("AutoLearningRate"));
    JUST(DoPass("QuantAwareTraining"));
    JUST(DoPass("GenerateBackwardAndOptimizerOpConfs"));
    JUST(DoPass("AddSspVariableProxy"));
    JUST(DoPass("CheckpointingPass"));
    JUST(DoPass("CudnnFusedNormalizationAddReluPass"));
    JUST(DoPass("PruneCastToStaticShapeOpsPass"));
    JUST(DoPass("FuseAddToOutputPass"));
    JUST(DoPass("IndexedSlicesOptimizerRewritePass"));
    JUST(DoPass("SplitSparseSoftmaxCrossEntropyOpPass"));
    JUST(DoPass("DoParallelCastBeforeWideningTypeCast"));
    JUST(DoPass("AddLbiDiffWatcherOpConfs"));
    JUST(DoPass("FuseCastScalePass"));
    JUST(DoPass("PruneParallelCastOpsPass"));
    JUST(DoPass("FuseUpdateOpsPass"));
    JUST(DoPass("DumpVariableInfoPass"));
  }
  JUST(DoPass("DumpTimeShapeAndBlobParallelConfPass"));
  return Maybe<void>::Ok();
}

Maybe<void> EagerJobBuildAndInferCtx::Complete() {
  CHECK_NOTNULL(Global<JobDesc>::Get());
  Global<JobDesc>::Delete();
  JUST(GetOpNames(job(), &executed_op_names_));
  auto scope = std::make_unique<GlobalJobDescScope>(mut_job()->job_conf(), job_id());
  JobPassCtx job_pass_ctx(GlobalJobDesc());
  auto DoPass = [&](const std::string& pass_name) -> Maybe<void> {
    return JobPass4Name(pass_name)(mut_job(), &job_pass_ctx);
  };
  JUST(DoPass("AutoTrainStep"));
  JUST(DoPass("AutoLearningRate"));
  JUST(DoPass("GenerateBackwardAndOptimizerOpConfs"));
  JUST(DoPass("AddLbiDiffWatcherOpConfs"));
  JUST(EagerRunOps(job(), &executed_op_names_, &ForeignCallback::EagerInterpretCompletedOp));
  return Maybe<void>::Ok();
}

void JobBuildAndInferCtx::InferBlobBackwardSignature(Operator* op) {
  std::function<bool(const LogicalBlobId&)> IsLbiBackwardUsed;
  InferBlobBackwardSignature(*op, &IsLbiBackwardUsed);
  auto* map = op->mut_blob_backward_used_signature()->mutable_bn_in_op2blob_backward_used();
  const auto& SetIsBlobBackwardUsed = [&](const std::string& bn_in_op) {
    (*map)[bn_in_op] = IsLbiBackwardUsed(op->BnInOp2Lbi(bn_in_op));
  };
  for (const auto& ibn : op->input_bns()) { SetIsBlobBackwardUsed(ibn); }
  for (const auto& obn : op->output_bns()) { SetIsBlobBackwardUsed(obn); }
}

void JobBuildAndInferCtx::InferBlobBackwardSignature(
    const Operator& op, std::function<bool(const LogicalBlobId&)>* IsLbiBackwardUsed) {
  if (op.job_desc().IsPredict()) {
    *IsLbiBackwardUsed = [](const LogicalBlobId&) { return false; };
    return;
  }
  const auto& Op4Name = [&](const std::string& op_name) { return CHECK_JUST(Op4OpName(op_name)); };
  UpdateOpName2AncestorsNeedNoGrad(op, Op4Name, &op_name2ancestors_need_no_grad_);
  // always return true if output_size > 1
  if (op.output_bns().size() > 1) {
    *IsLbiBackwardUsed = [](const LogicalBlobId&) { return true; };
    return;
  }
  std::vector<OperatorConf> bw_op_confs;
  LogicalBlobId fake_diff_lbi;
  fake_diff_lbi.set_op_name("fake_op_name");
  fake_diff_lbi.set_blob_name("fake_blob_name");
  HashMap<std::string, LogicalBlobId> in_diff2lbi;
  const auto& DiffLbi4BnInOp = [&](const std::string& bn) -> LogicalBlobId* {
    const auto& input_bns = op.input_bns();
    const auto& output_bns = op.output_bns();
    if (std::find(input_bns.begin(), input_bns.end(), bn) != input_bns.end()) {
      const auto& lbi = op.BnInOp2Lbi(bn);
      if (op_name2ancestors_need_no_grad_.at(lbi.op_name())) { return nullptr; }
      if (op.InputBlobModifier4Ibn(bn).requires_grad() == false) { return nullptr; }
      return &in_diff2lbi[bn];
    } else if (std::find(output_bns.begin(), output_bns.end(), bn) != output_bns.end()) {
      return &fake_diff_lbi;
    } else {
      LOG(FATAL) << "diff lbi for bn in op not found, bn: " << op.op_name() << "/" << bn;
    }
    return nullptr;
  };
  const auto& FwLogicalBlobDescPtr4Lbi = [&](const LogicalBlobId& lbi) -> const BlobDesc* {
    const auto& iter = lbi2logical_blob_desc_.find(lbi);
    if (iter != lbi2logical_blob_desc_.end()) { return iter->second.get(); }
    return nullptr;
  };
  const auto& LogicalBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
    const LogicalBlobId& lbi = op.BnInOp2Lbi(bn);
    const auto* logical_blob_desc = FwLogicalBlobDescPtr4Lbi(lbi);
    CHECK_NOTNULL(logical_blob_desc);
    return *logical_blob_desc;
  };
  const auto& maybe_ok =
      TRY(GenerateBackwardOpConfIf(op, &bw_op_confs, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp));
  CHECK(maybe_ok.IsOk() || maybe_ok.error()->has_gradient_function_not_found_error());
  // find backward used logical blob ids
  auto backward_used_lbis = std::make_shared<HashSet<LogicalBlobId>>();
  for (const auto& bw_op_conf : bw_op_confs) {
    const auto& bw_op = ConstructOp(bw_op_conf, op.device_type(), Global<JobDesc>::Get());
    for (const auto& ibn : bw_op->input_bns()) {
      const auto& lbi = bw_op->BnInOp2Lbi(ibn);
      if (FwLogicalBlobDescPtr4Lbi(lbi) != nullptr) { backward_used_lbis->insert(lbi); }
    }
  }
  *IsLbiBackwardUsed = [backward_used_lbis](const LogicalBlobId& lbi) {
    return backward_used_lbis->find(lbi) != backward_used_lbis->end();
  };
}

namespace {

std::string OpConf2ClassName(const OperatorConf& op_conf) {
  if (op_conf.has_user_conf()) {
    return op_conf.user_conf().op_type_name();
  } else if (op_conf.has_variable_conf()) {
    return "variable";
  } else if (op_conf.has_input_conf() && op_conf.has_return_conf()) {
    return "input";
  } else if (op_conf.has_output_conf() && op_conf.has_return_conf()) {
    return "output";
  } else {
    return "system_op";
  }
}

void FormateUserConf(nlohmann::json& json_conf) {
  nlohmann::json user_conf = json_conf["user_conf"];
  if (user_conf.is_null()) {
    json_conf.erase(json_conf.find("user_conf"));
    return;
  }
  std::string nomarl_array[] = {"at_int32",  "at_int64",  "at_bool",  "at_float",
                                "at_double", "at_string", "at_shape", "at_data_type"};
  std::string list_array[] = {"at_list_int32",     "at_list_int64", "at_list_float",
                              "at_list_data_type", "at_list_shape", "at_list_string"};
  nlohmann::json attr_json = user_conf["attr"];
  for (int32_t i = 0; i < attr_json.size(); i++) {
    std::string key = attr_json[i]["key"];
    nlohmann::json value_json = attr_json[i]["value"];
    bool is_found_normal = false;
    for (int32_t j = 0; j < nomarl_array->length(); j++) {
      std::string value_key = nomarl_array[j];
      if (value_json.contains(value_key)) {
        is_found_normal = true;
        if ("at_shape" == value_key) {
          json_conf[key] = value_json[value_key]["dim"];
        } else {
          json_conf[key] = value_json[value_key];
        }
        break;
      }
    }
    if (is_found_normal) { continue; }
    for (int32_t j = 0; j < list_array->length(); j++) {
      std::string value_key = list_array[j];
      if (value_json.contains(value_key)) {
        if (value_json[value_key].contains("val")) {
          json_conf[key] = value_json[value_key]["val"];
          break;
        } else if (value_json[value_key].contains("dim")) {
          json_conf[key] = value_json[value_key]["dim"];
          break;
        }
      }
    }
  }
  json_conf.erase(json_conf.find("user_conf"));
}

void FormateVariableConf(nlohmann::json& json_conf) {
  nlohmann::json variable_conf = json_conf["variable_conf"];
  if (variable_conf == nullptr) {
    json_conf.erase(json_conf.find("variable_conf"));
    return;
  }
  for (nlohmann::json::iterator it = variable_conf.begin(); it != variable_conf.end(); ++it) {
    std::string key = it.key();
    if ("shape" == key) {
      json_conf[key] = it.value()["dim"];
    } else {
      json_conf[key] = it.value();
    }
  }
  json_conf.erase(json_conf.find("variable_conf"));
}

}  // namespace

std::string oneflow::JobBuildAndInferCtx::GetJobStructureGraphJson(
    const std::string& job_name) const {
  HashSet<std::string> input_op_names;
  HashSet<std::string> output_op_names;
  std::vector<nlohmann::json> layers_vec;
  for (const auto& pair : op_name2op_) {
    nlohmann::json json_layers_pair;

    const Operator* op = pair.second.get();
    const std::string& op_name = pair.first;
    HashSet<std::string> inbound_nodes;
    for (const auto& ibn : op->input_bns()) {
      const LogicalBlobId& lbi = op->BnInOp2Lbi(ibn);
      if (op_name2op_.find(lbi.op_name()) != op_name2op_.end()) {
        inbound_nodes.insert(lbi.op_name());
      }
    }

    if (op->op_conf().has_input_conf() && op->op_conf().has_return_conf()) {
      input_op_names.insert(op_name);
    }
    if (op->op_conf().has_output_conf() && op->op_conf().has_return_conf()) {
      output_op_names.insert(op_name);
    }
    json_layers_pair["name"] = op_name;

    std::string class_name = OpConf2ClassName(op->op_conf());
    json_layers_pair["class_name"] = class_name;

    nlohmann::json json_conf;
    summary::ConvertProtobufMsg2Json(json_conf, op->op_conf());
    FormateUserConf(json_conf);
    FormateVariableConf(json_conf);
    json_layers_pair["config"] = json_conf;

    std::vector<std::string> inbound_nodes_vec;
    for (const auto& in_node_name : inbound_nodes) { inbound_nodes_vec.emplace_back(in_node_name); }
    json_layers_pair["inbound_nodes"] = inbound_nodes_vec;

    layers_vec.emplace_back(json_layers_pair);
  }

  nlohmann::json json_pair;
  json_pair["name"] = job_name;
  json_pair["layers"] = layers_vec;
  json_pair["input_layers"] = input_op_names;
  json_pair["output_layers"] = output_op_names;

  return json_pair.dump();
}

Maybe<void> JobBuildAndInferCtx::Rebuild() {
  // clear old state
  lbi2logical_blob_desc_.clear();
  lbi2sbp_parallel_from_producer_view_.clear();
  lbi2parallel_desc_from_producer_view_.clear();
  lbi2disable_boxing_.clear();
  op_name2op_.clear();
  parallel_desc2placement_group_.clear();
  parallel_desc2blob_placement_group_.clear();
  consistent_lbi2mirrored_lbi_.clear();
  mirrored_lbi2sub_lbis_.clear();
  mirrored_lbi2parallel_desc_.clear();
  mirrored_lbi2sbp_parallel_.clear();
  op_name2ancestors_need_no_grad_.clear();
  // record op mirror view
  HashMap<std::string, bool> op_name2is_mirrored;
  CHECK_OR_RETURN(job_->has_job_parallel_view_conf());
  for (const auto& op_conf : job_->net().op()) {
    const auto& op_name = op_conf.name();
    CHECK_OR_RETURN(op_name2is_mirrored.find(op_name) == op_name2is_mirrored.end());
    op_name2is_mirrored[op_name] = false;
    const auto& op_name2is_mirrored_parallel_view =
        job_->job_parallel_view_conf().op_name2is_mirrored_parallel_view();
    if (op_name2is_mirrored_parallel_view.find(op_name)
        != op_name2is_mirrored_parallel_view.end()) {
      if (op_name2is_mirrored_parallel_view.at(op_name)) { op_name2is_mirrored[op_name] = true; }
    }
  }
  // build op graph
  OpGraph op_graph;
  if (Global<JobDesc>::Get()) {
    op_graph.Init(*job_);
  } else {
    auto scope = std::make_unique<GlobalJobDescScope>(job_->job_conf(), job_id());
    op_graph.Init(*job_);
  }
  // clear old job except job_conf
  job_->mutable_net()->Clear();
  job_->mutable_placement()->Clear();
  job_->mutable_job_parallel_view_conf()->Clear();
  job_->mutable_helper()->Clear();
  // topo traverse op_graph to AddAndInferOp
  op_graph.TopoForEachNode([&](OpNode* node) -> void {
    const auto& op_conf = node->op().op_conf();
    CHECK(op_name2is_mirrored.find(op_conf.name()) != op_name2is_mirrored.end());
    bool is_mirrored = op_name2is_mirrored.at(op_conf.name());
    if (is_mirrored) {
      CHECK_JUST(AddAndInferMirroredOp(op_conf));
    } else {
      CHECK_JUST(AddAndInferConsistentOp(op_conf));
    }
  });
  // updata job_helper
  op_graph.DumpOpTimeShape(job_);
  op_graph.DumpLogicalBlobDesc(job_);
  op_graph.DumpSbpSignature(job_);
  return Maybe<void>::Ok();
}

Maybe<std::string> JobBuildAndInferCtx::GetOpBlobLbn(const std::string& op_name,
                                                     const std::string& bn_in_op) const {
  const auto& lbi = JUST(Op4OpName(op_name))->BnInOp2Lbi(bn_in_op);
  return GenLogicalBlobName(lbi);
}

}  // namespace oneflow
