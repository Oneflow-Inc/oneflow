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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/cost_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/local_sig_infer_hint.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/user/summary/summary_converter.h"

#include <google/protobuf/text_format.h>
#include "nlohmann/json.hpp"

namespace oneflow {

static const std::string kAutoLocalBlobNamePrefix =
    "System-Local-Blob-Auto-Converted-From-Global-Blob";

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

void UpdateOpName2AncestorsNeedNoGrad(
    const Operator& op, const std::function<const Operator*(const std::string&)>& Op4OpName,
    const bool is_train, HashMap<std::string, bool>* op_name2ancestors_need_no_grad) {
  bool no_grad = !is_train;
  auto IsTrainableVariableLbi = [&](const LogicalBlobId& lbi) {
    const auto& op_conf = Op4OpName(lbi.op_name())->op_conf();
    return op_conf.has_variable_conf() && op_conf.variable_conf().trainable();
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

JobBuildAndInferCtx::JobBuildAndInferCtx(Job* job, int64_t job_id)
    : job_(job), job_id_(job_id), unique_op_name_index_(0) {
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
  CHECK_ISNULL_OR_RETURN(Singleton<JobDesc>::Get());
  Singleton<JobDesc>::New(job_conf, job_id_);
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
    *iter->second->add_lbi() = lbi;
  }
  return Maybe<void>::Ok();
}

Maybe<OperatorConf> JobBuildAndInferCtx::DecodeLbiHintAndReturnNewOpConf(
    const Operator& op, SbpSignature* sbp_sig_conf) const {
  auto op_conf_without_split_hint = std::make_shared<OperatorConf>(op.op_conf());
  for (const std::string& ibn : op.input_bns()) {
    std::string lbn_may_with_hint = GetInputLbnInOpCustomizedConf(op.op_conf(), ibn);
    SbpParallel sbp_parallel;
    bool has_sbp_hint = JUST(GetSbpParallelInLbnOrNothing(lbn_may_with_hint, &sbp_parallel));
    if (has_sbp_hint) {
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
                                                            const ParallelDesc& parallel_desc,
                                                            const NdSbpSignature& nd_sbp_signature,
                                                            bool is_local_parallel_view) const {
  auto* op_name2sbp_sig =
      job_->mutable_job_parallel_view_conf()->mutable_op_name2sbp_signature_conf();
  auto* op_name2nd_sbp_sig =
      job_->mutable_job_parallel_view_conf()->mutable_op_name2nd_sbp_signature_conf();
  if (nd_sbp_signature.bn_in_op2nd_sbp().size() > 0) {
    (*op_name2nd_sbp_sig)[operator_conf.name()] = nd_sbp_signature;
    if (parallel_desc.hierarchy()->NumAxes() == 1) {
      SbpSignature sbp_signature;
      NdSbpSignatureToSbpSignature(nd_sbp_signature, &sbp_signature);
      (*op_name2sbp_sig)[operator_conf.name()] = sbp_signature;
    }
  }
  auto* op_name2is_local_parallel_view =
      job_->mutable_job_parallel_view_conf()->mutable_op_name2is_local_parallel_view();
  if (is_local_parallel_view) { (*op_name2is_local_parallel_view)[operator_conf.name()] = true; }
  job_->mutable_net()->add_op()->CopyFrom(operator_conf);

  // set up the module config
  const auto& scope =
      Singleton<symbol::Storage<Scope>>::Get()->Get(operator_conf.scope_symbol_id());
  if (scope.scope_proto().has_module_name()) {
    const auto& module_name = scope.scope_proto().module_name();
    auto* module_name2module_conf = job_->mutable_module_name2module_conf();
    if (!(*module_name2module_conf)[module_name].has_name()) {
      (*module_name2module_conf)[module_name].set_name(scope.scope_proto().module_name());
    }

    *((*module_name2module_conf)[module_name].add_ops()) = operator_conf.name();
  }
}

Maybe<void> JobBuildAndInferCtx::InferLocalSignature(Operator* op, bool is_local_parallel_view_conf,
                                                     const ParallelDesc& parallel_desc) {
  HashMap<std::string, LocalSigInferHint> ibn2local_sig_infer_hint;
  for (const std::string& ibn : op->input_bns()) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(ibn);
    CHECK_OR_RETURN(lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end())
        << Error::LogicalBlobNameNotExistError()
        << "infer blob desc not found, when infer op_name: \"" << op->op_name()
        << "\", consumed op_name: \"" << lbi.op_name() << "\", blob_name: \"" << lbi.blob_name();
    const ParallelDesc* pd = &lbi2parallel_desc_from_producer_view_.at(lbi);
    const auto* producer_op = op_name2op_.at(lbi.op_name()).get();
    const auto& producer_obn = *JUST(producer_op->obn4lbi(lbi));
    const auto& opt_local_parallel =
        *CHECK_JUST(producer_op->OptLocalParallel4BnInOp(producer_obn));
    ibn2local_sig_infer_hint.emplace(
        ibn, LocalSigInferHint(pd, opt_local_parallel.has_local_parallel()));
  }
  const auto& LocalSigInferHint4Ibn =
      [&](const std::string& ibn) -> Maybe<const LocalSigInferHint*> {
    const auto& iter = ibn2local_sig_infer_hint.find(ibn);
    CHECK_OR_RETURN(iter != ibn2local_sig_infer_hint.end()) << "input blob not found. ibn: " << ibn;
    return &iter->second;
  };
  JUST(
      op->InferLocalSignatureIf(LocalSigInferHint4Ibn, is_local_parallel_view_conf, parallel_desc));
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::InferOpOutNdSbp(Operator* op,
                                                 const NdSbpSignature& nd_sbp_sig_conf,
                                                 const ParallelDesc& parallel_desc) {
  HashMap<std::string, NdSbpInferHint> ibn2nd_sbp_infer_hint;
  for (const std::string& ibn : op->input_bns()) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(ibn);
    auto logical_blob_desc_it = lbi2logical_blob_desc_.find(lbi);
    CHECK_OR_RETURN(logical_blob_desc_it != lbi2logical_blob_desc_.end())
        << Error::LogicalBlobNameNotExistError()
        << "infer blob desc not found, when infer op_name: \"" << op->op_name()
        << "\", consumed op_name: \"" << lbi.op_name() << "\", blob_name: \"" << lbi.blob_name();
    const BlobDesc* logical_blob_desc = logical_blob_desc_it->second.get();
    const ParallelDesc* pd = &lbi2parallel_desc_from_producer_view_.at(lbi);
    auto nd_sbp_it = lbi2nd_sbp_from_producer_view_.find(lbi);
    CHECK_OR_RETURN(nd_sbp_it != lbi2nd_sbp_from_producer_view_.end())
        << Error::LogicalBlobNameNotExistError() << "when infer op_name: " << op->op_name()
        << " consumed op_name: " << lbi.op_name() << " blob_name: " << lbi.blob_name()
        << " not infer parallel distribution";
    const NdSbp* nd_sbp = &nd_sbp_it->second;
    ibn2nd_sbp_infer_hint.emplace(ibn, NdSbpInferHint(pd, logical_blob_desc, nd_sbp));
  }

  const auto NdSbpInferHint4Ibn = [&](const std::string& bn) -> Maybe<const NdSbpInferHint*> {
    return &ibn2nd_sbp_infer_hint.at(bn);
  };

  JUST(op->InferNdSbpSignatureIf(nd_sbp_sig_conf, parallel_desc, NdSbpInferHint4Ibn));

  const auto& bn2nd_sbp = JUST(op->nd_sbp_signature())->bn_in_op2nd_sbp();
  for (const auto& obn : op->output_bns()) {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(obn);
    CHECK_OR_RETURN(bn2nd_sbp.find(obn) != bn2nd_sbp.end())
        << Error::BlobSplitAxisInferError() << "op_name: " << lbi.op_name()
        << " blob_name: " << lbi.blob_name() << " not infer split axis";
    CHECK_OR_RETURN(lbi2nd_sbp_from_producer_view_.emplace(lbi, bn2nd_sbp.at(obn)).second)
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
  produced_bns.reserve(op->output_bns().size() + op->tmp_bns().size());
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
  const auto& parallel_hierarchy = JUST(op->GetOpParallelDesc())->hierarchy();
  if (parallel_hierarchy->NumAxes() == 1) {
    HashSet<std::string> obns(op->output_bns().begin(), op->output_bns().end());
    auto GetParallelNum = [&](const std::string& bn_in_op) {
      if (obns.find(bn_in_op) == obns.end()) { return parallel_num; }
      return lbi2parallel_desc_from_producer_view_.at(op->BnInOp2Lbi(bn_in_op)).parallel_num();
    };
    for (const auto& pair : JUST(op->sbp_signature())->bn_in_op2sbp_parallel()) {
      if (!pair.second.has_split_parallel()) { continue; }
      if (JUST(op->OptLocalParallel4BnInOp(pair.first))->has_local_parallel()) { continue; }
      int64_t axis = pair.second.split_parallel().axis();
      const LogicalBlobId& lbi = op->BnInOp2Lbi(pair.first);
      int64_t blob_parallel_num = GetParallelNum(pair.first);
      const BlobDesc& logical_blob_desc = *(lbi2logical_blob_desc_.at(lbi).get());
      int64_t num_axes = logical_blob_desc.shape().NumAxes();
      if (axis < 0) { axis += num_axes; }
      CHECK_GE_OR_RETURN(axis, 0);
      CHECK_LE_OR_RETURN(axis, num_axes)
          << "op: " << op->op_name() << ", blob: " << pair.first << ", axis: " << axis
          << ", shape: " << logical_blob_desc.shape();
      if (logical_blob_desc.shape().NumAxes() > 0) {
        CHECK_GE_OR_RETURN(logical_blob_desc.shape().At(axis), blob_parallel_num)
            << "op_name: " << lbi.op_name() << " blob_name: " << lbi.blob_name()
            << " shape: " << logical_blob_desc.shape()
            << " cannot be splitted by parallel_num: " << blob_parallel_num << " at axis " << axis;
      }
    }
  } else {
    for (const auto& pair : JUST(op->nd_sbp_signature())->bn_in_op2nd_sbp()) {
      if (JUST(op->OptLocalParallel4BnInOp(pair.first))->has_local_parallel()) { continue; }
      const LogicalBlobId& lbi = op->BnInOp2Lbi(pair.first);
      const BlobDesc& logical_blob_desc = *(lbi2logical_blob_desc_.at(lbi).get());
      Shape current_shape = logical_blob_desc.shape();
      for (int64_t i = 0; i < pair.second.sbp_parallel_size(); ++i) {
        const SbpParallel& sbp_parallel = pair.second.sbp_parallel(i);
        if (sbp_parallel.has_split_parallel()) {
          const int64_t axis = sbp_parallel.split_parallel().axis();
          CHECK_GT_OR_RETURN(current_shape.At(axis), 0);
          // Support unbalanced splitting
          CHECK_GE_OR_RETURN(current_shape.At(axis), parallel_hierarchy->At(i))
              << "op_name: " << lbi.op_name() << " blob_name: " << lbi.blob_name()
              << " shape: " << logical_blob_desc.shape()
              << " cannot be splitted by nd sbp: " << NdSbpToString(pair.second) << " at axis "
              << axis << " with parallel_hierarchy: " << *parallel_hierarchy;
          // Split and take the minimum one
          current_shape.Set(axis, current_shape.At(axis) / parallel_hierarchy->At(i));
        }
      }
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

Maybe<NdSbpSignature> JobBuildAndInferCtx::InitConstraitNdSbpSignature(
    const Operator& op, const HashMap<std::string, bool>& ibn2disable_boxing) const {
  auto nd_sbp_sig = std::make_shared<NdSbpSignature>();
  for (const auto& it : ibn2disable_boxing) {
    if (it.second) {
      const auto& ibn = it.first;
      const LogicalBlobId& lbi = op.BnInOp2Lbi(ibn);
      const auto& nd_sbp_iter = lbi2nd_sbp_from_producer_view_.find(lbi);
      if (nd_sbp_iter == lbi2nd_sbp_from_producer_view_.end()) {
        return Error::RuntimeError()
               << "The nd_sbp of input " << ibn << " (tensor name is " << GenLogicalBlobName(lbi)
               << ") is not found for operation " << op.op_name()
               << ". It maybe caused by an invalid inplace operation.";
      }
      (*(nd_sbp_sig->mutable_bn_in_op2nd_sbp()))[ibn] = lbi2nd_sbp_from_producer_view_.at(lbi);
    }
  }
  return nd_sbp_sig;
}

bool JobBuildAndInferCtx::HasAnyLocalBlobInput(const Operator& op) const {
  for (const auto& ibn : op.input_bns()) {
    const auto& lbi = op.BnInOp2Lbi(ibn);
    if (local_lbi2sub_lbis_.find(lbi) != local_lbi2sub_lbis_.end()) { return true; }
  }
  return false;
}

Maybe<const SbpParallel*> JobBuildAndInferCtx::SbpParallel4Lbi(const LogicalBlobId& lbi) const {
  const auto& iter = lbi2nd_sbp_from_producer_view_.find(lbi);
  CHECK_OR_RETURN(iter != lbi2nd_sbp_from_producer_view_.end())
      << "lbn: " << GenLogicalBlobName(lbi) << " undefined";
  CHECK_EQ_OR_RETURN(iter->second.sbp_parallel_size(), 1);
  return &(iter->second.sbp_parallel(0));
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
    const auto& iter = local_lbi2sbp_parallel_.find(lbi);
    if (iter != local_lbi2sbp_parallel_.end()) {
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

Maybe<void> JobBuildAndInferCtx::CheckAllInputsConvertableToLocalBlob(const Operator& op) const {
  for (const auto& ibn : op.input_bns()) {
    const auto& lbi = op.BnInOp2Lbi(ibn);
    if (local_lbi2sub_lbis_.find(lbi) != local_lbi2sub_lbis_.end()) { continue; }
    const auto& sbp = *JUST(SbpParallel4Lbi(lbi));
    if (sbp.has_broadcast_parallel()) { continue; }
    if (sbp.has_split_parallel() && sbp.split_parallel().axis() == 0) { continue; }
    const std::string& lbn = GenLogicalBlobName(lbi);
    return Error::CheckFailedError() << "input lbn: " << lbn << " is not convertable to local blob";
  }
  return Maybe<void>::Ok();
}

Maybe<void> LazyJobBuildAndInferCtx::CheckAllInputsWithSameParallelNum(const Operator& op,
                                                                       int32_t parallel_num) const {
  for (const auto& ibn : op.input_bns()) {
    const auto& lbi = op.BnInOp2Lbi(ibn);
    const auto& iter = local_lbi2sub_lbis().find(lbi);
    int32_t ibn_parallel_num = 0;
    if (iter != local_lbi2sub_lbis().end()) {
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

Maybe<OpAttribute> JobBuildAndInferCtx::AddAndInferLocalOp(const OperatorConf& op_conf) {
  CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
  const auto& scope = Singleton<symbol::Storage<Scope>>::Get()->Get(op_conf.scope_symbol_id());
  const auto* job_desc = JUST(scope.job_desc());
  const auto& parallel_desc = *JUST(scope.GetParallelDesc(op_conf));
  auto op = JUST(ConstructOp(op_conf, parallel_desc.device_type()));
  JUST(CheckAllInputsConvertableToLocalBlob(*op));
  int32_t parallel_num = parallel_desc.parallel_num();
  JUST(CheckAllInputsWithSameParallelNum(*op, parallel_num));
  auto GetSubOpName = [&](int index) { return GetLocalOpName(op_conf.name(), index); };
  OperatorConf sub_op_conf(op_conf);
  int64_t sub_op_list_size = SizeOfSubGlobalOpList(parallel_num);
  auto last_op_attribute = std::make_shared<OpAttribute>();
  FOR_RANGE(int32_t, i, 0, sub_op_list_size) {
    ResetOpConfName(&sub_op_conf, GetSubOpName(i));
    for (const auto& ibn : op->input_bns()) {
      const auto& lbi = *JUST(GetSubLbi(op_conf.scope_symbol_id(), op->BnInOp2Lbi(ibn), i));
      ReplaceInputLbnInOpCustomizedConf(&sub_op_conf, ibn, GenLogicalBlobName(lbi));
    }
    const ParallelConf& parallel_conf = GetLocalOpParallelConf(parallel_desc, i);
    bool is_local_parallel_view = GetIsLocalParallelView();
    last_op_attribute =
        JUST(AddAndInferOp(sub_op_conf, parallel_conf, job_desc, is_local_parallel_view));
  }
  bool is_broadcast = JUST(AllInputsBroadcastParallel(*op));
  for (const auto& obn : op->output_bns()) {
    const auto& lbi = op->BnInOp2Lbi(obn);
    auto* sub_lbis = &local_lbi2sub_lbis_[lbi];
    sub_lbis->resize(sub_op_list_size, op->BnInOp2Lbi(obn));
    FOR_RANGE(int32_t, i, 0, sub_op_list_size) { sub_lbis->at(i).set_op_name(GetSubOpName(i)); }
    CHECK(local_lbi2parallel_desc_.emplace(lbi, parallel_desc).second);
    auto* sbp_parallel = &local_lbi2sbp_parallel_[lbi];
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
  auto lbi_vec_iter = local_lbi2sub_lbis_.find(lbi);
  if (lbi_vec_iter == local_lbi2sub_lbis_.end()) {
    const auto& new_lbi = JUST(FindOrCreateLocalLbiFromCompatibleGlobalBlob(scope_symbol_id, lbi));
    lbi_vec_iter = local_lbi2sub_lbis_.find(*new_lbi);
    CHECK(lbi_vec_iter != local_lbi2sub_lbis_.end());
  }
  return &lbi_vec_iter->second.at(index);
}

Maybe<OpAttribute> JobBuildAndInferCtx::AddAndInferGlobalOp(const OperatorConf& op_conf) {
  CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
  const auto& scope = Singleton<symbol::Storage<Scope>>::Get()->Get(op_conf.scope_symbol_id());
  const auto& parallel_desc = *JUST(scope.GetParallelDesc(op_conf));
  const auto* job_desc = JUST(scope.job_desc());
  return AddAndInferOp(op_conf, parallel_desc.parallel_conf(), job_desc, false);
}

// TODO(): add handle error of same interface op blob between jobs
Maybe<OpAttribute> JobBuildAndInferCtx::AddAndInferOp(const OperatorConf& op_conf,
                                                      const ParallelConf& origin_parallel_conf,
                                                      const JobDesc* job_desc,
                                                      bool is_local_parallel_view) {
  CHECK_OR_RETURN(has_job_conf_) << Error::JobConfNotSetError();
  if (!is_job_conf_frozen_) { is_job_conf_frozen_ = true; }
  const std::string& op_name = op_conf.name();
  CHECK_OR_RETURN(op_name2op_.find(op_name) == op_name2op_.end())
      << Error::OpNameExistError() << "op_name: " << op_name
      << " already exist in job: " << job_->job_conf().job_name();
  CHECK_NE_OR_RETURN(op_conf.device_tag(), "invalid_device")
      << Error::OpConfDeviceTagNoSetError() << "op_name: " << op_name << " not set device tag";

  op_name2op_.emplace(op_name, JUST(ConstructOp(op_conf)));
  Operator* op = op_name2op_.at(op_name).get();

  SbpSignature sbp_sig_conf;
  HashMap<std::string, bool> ibn2disable_boxing;
  InitIbn2DisableBoxing(*op, &ibn2disable_boxing);
  auto new_op_conf = JUST(DecodeLbiHintAndReturnNewOpConf(*op, &sbp_sig_conf));
  auto parallel_conf = JUST(InferOpParallelConf(*op, origin_parallel_conf, ibn2disable_boxing));
  ParallelDesc parallel_desc(*parallel_conf);
  JUST(op->FillOpParallelDesc(parallel_desc));
  JUST(AddOpNameParallelConf2Placement(op_name, *parallel_conf));

  auto GetBlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
    const LogicalBlobId& lbi = op->BnInOp2Lbi(bn);
    if (lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end()) {
      return lbi2logical_blob_desc_.at(lbi).get();
    }
    return nullptr;
  };
  JUST(op->FillLogicalInBlobDesc(GetBlobDesc4BnInOp));
  JUST(op->InferParallelSignatureIf());

  // infer local signature
  JUST(InferLocalSignature(op, is_local_parallel_view, parallel_desc));

  // infer nd_sbp signature
  NdSbpSignature nd_sbp_sig_conf;
  // Only infer nd_sbp signature if auto parallel is not enable,
  // since the semi-auto parallellism rule might have inconsistency with the auto-parallel strategy.
  if (!job_desc->enable_auto_parallel()) {
    nd_sbp_sig_conf = *JUST(InitConstraitNdSbpSignature(*op, ibn2disable_boxing));
  }
  // Override constrait nd_sbp if sbp hint is given
  if (!sbp_sig_conf.bn_in_op2sbp_parallel().empty()) {
    SbpSignatureToNdSbpSignature(sbp_sig_conf, &nd_sbp_sig_conf);
  }
  AddOpAndUpdateJobParallelViewConf(*new_op_conf, parallel_desc, nd_sbp_sig_conf,
                                    is_local_parallel_view);
  JUST(InferOpOutNdSbp(op, nd_sbp_sig_conf, parallel_desc));

  // infer logical blob desc
  JUST(GenOpProducedEmptyLogicalBlobDesc(op));
  JUST(op->InferLogicalOutBlobDescsIf());
  for (const auto& bn : op->output_bns()) {
    *lbi2logical_blob_desc_.at(op->BnInOp2Lbi(bn)) = *JUST(op->GetLogicalBlobDesc4Obn(bn));
  }
  // Infer ParallelDesc for output blobs.
  auto ParallelDesc4Obn = [&](const std::string& obn) -> ParallelDesc* {
    const auto& lbi = op->BnInOp2Lbi(obn);
    auto iter = lbi2parallel_desc_from_producer_view_.find(lbi);
    if (iter == lbi2parallel_desc_from_producer_view_.end()) {
      iter = lbi2parallel_desc_from_producer_view_.emplace(lbi, parallel_desc).first;
    }
    return &iter->second;
  };
  for (const auto& bn : op->output_bns()) {
    lbi2parallel_desc_from_producer_view_.emplace(op->BnInOp2Lbi(bn),
                                                  *JUST(op->GetParallelDesc4BnInOp(bn)));
  }
  JUST(AddLbiParallelConf2BlobPlacement(op, ParallelDesc4Obn));
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
  if (IsLocalBlob(lbn)) { return AddLossLocalBlobName(lbn); }
  return AddLossGlobalBlobName(lbn);
}

Maybe<void> JobBuildAndInferCtx::AddLossGlobalBlobName(const std::string& lbn) {
  JUST(CheckLbnValidAndExist(lbn));
  CHECK_OR_RETURN(job_->job_conf().has_train_conf())
      << Error::UnknownJobBuildAndInferError()
      << "job has no TrainConf when adding loss logical blob name";
  job_->mutable_job_conf()->mutable_train_conf()->add_loss_lbn(lbn);
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::MarkVariableGradientBlobNames(
    const HashMap<std::string, std::string>& variable_grad_lbns) {
  CHECK_OR_RETURN(job_->job_conf().has_train_conf())
      << Error::UnknownJobBuildAndInferError()
      << "job has no TrainConf when add variable gradient logical blob name";
  auto* train_conf = job_->mutable_job_conf()->mutable_train_conf();
  for (int i = 0; i < train_conf->optimizer_conf_size(); ++i) {
    auto* optimizer_conf = train_conf->mutable_optimizer_conf(i);
    for (const auto& variable_op_name : optimizer_conf->variable_op_names()) {
      const auto& it = variable_grad_lbns.find(variable_op_name + "/out");
      if (it != variable_grad_lbns.end()) {
        optimizer_conf->add_variable_grad_lbns(it->second);
      } else {
        // add an empty gradient lbn for variable that has no gradient
        optimizer_conf->add_variable_grad_lbns("");
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtx::MarkOutputGradientBlobNames(
    const HashMap<std::string, std::string>& output_gradient_lbns) {
  CHECK_OR_RETURN(job_->job_conf().has_train_conf())
      << Error::UnknownJobBuildAndInferError()
      << "job has no TrainConf when add variable gradient logical blob name";
  auto* train_conf = job_->mutable_job_conf()->mutable_train_conf();
  for (const auto& loss_lbn : train_conf->loss_lbn()) {
    const auto& it = output_gradient_lbns.find(loss_lbn);
    CHECK_OR_RETURN(it != output_gradient_lbns.end())
        << Error::UnknownJobBuildAndInferError() << "gradient is missing for loss " << loss_lbn;
    train_conf->add_loss_grad_lbn(it->second);
  }
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

Maybe<bool> JobBuildAndInferCtx::IsDisableBoxing(const std::string& lbn) const {
  JUST(CheckLbnValidAndExist(lbn));
  LogicalBlobId lbi(GenLogicalBlobId(lbn));
  const auto& iter = lbi2disable_boxing_.find(lbi);
  CHECK_OR_RETURN(iter != lbi2disable_boxing_.end());
  return iter->second;
}

Maybe<void> JobBuildAndInferCtx::DisableBoxing(const std::string& lbn) {
  JUST(CheckLbnValidAndExist(lbn));
  LogicalBlobId lbi(GenLogicalBlobId(lbn));
  lbi2disable_boxing_[lbi] = true;
  return Maybe<void>::Ok();
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
  const auto& nd_sbp = lbi2nd_sbp_from_producer_view_.at(GenLogicalBlobId(lbn));
  CHECK_EQ_OR_RETURN(nd_sbp.sbp_parallel_size(), 1);
  const auto& sbp = nd_sbp.sbp_parallel(0);
  if (sbp.has_split_parallel()) { ret.set_value(sbp.split_parallel().axis()); }
  return ret;
}

Maybe<const ParallelDesc*> JobBuildAndInferCtx::GetParallelDescFromProducerView(
    const std::string& lbn) const {
  JUST(CheckLbnValidAndExist(lbn));
  return &(lbi2parallel_desc_from_producer_view_.at(GenLogicalBlobId(lbn)));
}

Maybe<void> JobBuildAndInferCtx::AddLossLocalBlobName(const std::string& lbn) {
  const auto& local_lbi = JUST(GetLocalLbi(lbn));
  CHECK_OR_RETURN(job_->job_conf().has_train_conf())
      << Error::UnknownJobBuildAndInferError()
      << "job has no TrainConf when adding loss logical blob name";
  for (const auto& lbi : local_lbi2sub_lbis_[*local_lbi]) {
    job_->mutable_job_conf()->mutable_train_conf()->add_loss_lbn(GenLogicalBlobName(lbi));
  }
  return Maybe<void>::Ok();
}

Maybe<LogicalBlobId> JobBuildAndInferCtx::GetLocalLbi(const std::string& lbn_with_hint) const {
  const LogicalBlobId& lbi = GenLogicalBlobId(lbn_with_hint);
  if (local_lbi2sub_lbis_.find(lbi) != local_lbi2sub_lbis_.end()) { return lbi; }
  return Error::CheckFailedError() << lbn_with_hint << " is not a local blob name";
}

Maybe<int> JobBuildAndInferCtx::LocalBlobGetNumSubLbi(const std::string& lbn_with_hint) const {
  const auto& local_lbi = JUST(GetLocalLbi(lbn_with_hint));
  return local_lbi2sub_lbis_.at(*local_lbi).size();  // NOLINT
}

Maybe<const LogicalBlobId*> JobBuildAndInferCtx::LocalBlobGetSubLbi(
    const std::string& lbn_with_hint, int index) const {
  const auto& local_lbi = JUST(GetLocalLbi(lbn_with_hint));
  const auto& vec = local_lbi2sub_lbis_.at(*local_lbi);  // NOLINT
  CHECK_GE_OR_RETURN(index, 0);
  CHECK_LT_OR_RETURN(index, vec.size());
  return &vec.at(index);
}

bool JobBuildAndInferCtx::IsLocalBlob(const std::string& lbn) const {
  bool is_local_blob = TRY(GetLocalLbi(lbn)).IsOk();
  if (is_local_blob) { return is_local_blob; }
  const LogicalBlobId& lbi = GenLogicalBlobId(lbn);
  CHECK(lbi2logical_blob_desc_.find(lbi) != lbi2logical_blob_desc_.end()) << "lbn: " << lbn;
  return false;
}

Maybe<Shape> JobBuildAndInferCtx::LocalBlobGetStaticShape(const std::string& lbn_with_hint) const {
  const auto& lbi = *JUST(LocalBlobGetSubLbi(lbn_with_hint, 0));
  return lbi2logical_blob_desc_.at(lbi)->shape();
}

Maybe<DataType> JobBuildAndInferCtx::LocalBlobGetDataType(const std::string& lbn_with_hint) const {
  const auto& lbi = *JUST(LocalBlobGetSubLbi(lbn_with_hint, 0));
  return lbi2logical_blob_desc_.at(lbi)->data_type();
}

Maybe<bool> JobBuildAndInferCtx::LocalBlobIsDynamic(const std::string& lbn_with_hint) const {
  const auto& lbi = *JUST(LocalBlobGetSubLbi(lbn_with_hint, 0));
  return lbi2logical_blob_desc_.at(lbi)->is_dynamic();
}

Maybe<OptInt64> JobBuildAndInferCtx::LocalBlobGetSplitAxisFromProducerView(
    const std::string& lbn_with_hint) const {
  const auto& lbi = *JUST(LocalBlobGetSubLbi(lbn_with_hint, 0));
  OptInt64 ret;
  const auto& nd_sbp = lbi2nd_sbp_from_producer_view_.at(lbi);
  CHECK_EQ_OR_RETURN(nd_sbp.sbp_parallel_size(), 1);
  const auto& sbp = nd_sbp.sbp_parallel(0);
  if (sbp.has_split_parallel()) { ret.set_value(sbp.split_parallel().axis()); }
  return ret;
}

Maybe<const ParallelDesc*> JobBuildAndInferCtx::LocalBlobGetParallelDescFromProducerView(
    const std::string& lbn_with_hint) const {
  const auto& lbi = JUST(GetLocalLbi(lbn_with_hint));
  return &(local_lbi2parallel_desc_.at(*lbi));  // NOLINT
}

Maybe<void> JobBuildAndInferCtx::CheckJob() const {
  JUST(CheckPlacement());
  JUST(CheckJobConf());
  JUST(CheckOpScope());
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

Maybe<void> JobBuildAndInferCtx::CheckOpScope() const {
  for (const OperatorConf& op_conf : job_->net().op()) {
    if (!op_conf.has_scope_symbol_id()) {
      // NOTE(chengcheng): LOG(WARNING) instead of CHECK_OR_RETURN() for transition
      LOG(WARNING) << " ERROR! op_name: " << op_conf.name()
                   << " has NOT set scope(scope_symbol_id) in job: " << job_->job_conf().job_name()
                   << " net. \n op_conf = " << op_conf.DebugString();
    }
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
  CHECK_HAS_LBI_KEY(lbi2nd_sbp_from_producer_view_);
  CHECK_HAS_LBI_KEY(lbi2parallel_desc_from_producer_view_);
#undef CHECK_HAS_LBI_KEY

  return Maybe<void>::Ok();
}

const Job& JobBuildAndInferCtx::job() const { return *job_; }

std::string LazyJobBuildAndInferCtx::GetLocalOpName(const std::string& op_name,
                                                    int64_t parallel_id) const {
  return op_name + "_" + std::to_string(parallel_id);
}

ParallelConf LazyJobBuildAndInferCtx::GetLocalOpParallelConf(const ParallelDesc& parallel_desc,
                                                             int64_t parallel_id) const {
  return parallel_desc.GetParallelIdOnlyParallelConf(parallel_id);
}

Maybe<LogicalBlobId> LazyJobBuildAndInferCtx::FindOrCreateLocalLbiFromCompatibleGlobalBlob(
    int64_t scope_symbol_id, const LogicalBlobId& lbi) {
  const std::string& lbn = GenLogicalBlobName(lbi);
  const auto& sbn_it = mut_global_lbi2local_lbi()->find(lbi);
  if (sbn_it != mut_global_lbi2local_lbi()->end()) { return sbn_it->second; }
  const SbpParallel& sbp = *JUST(SbpParallel4Lbi(lbi));
  const ParallelDesc& parallel_desc = *JUST(ParallelDesc4Lbi(lbi));
  LogicalBlobId local_lbi;
  local_lbi.set_op_name(kAutoLocalBlobNamePrefix + NewUniqueId());
  local_lbi.set_blob_name("out");
  (*mut_global_lbi2local_lbi())[lbi] = local_lbi;
  auto* lbi_vec = &(*mut_local_lbi2sub_lbis())[local_lbi];
  lbi_vec->reserve(parallel_desc.parallel_num());
  auto PushBackSubLbi = [&](const std::string& op_name, const std::string& blob_name) {
    LogicalBlobId sub_lbi;
    sub_lbi.set_op_name(op_name);
    sub_lbi.set_blob_name(blob_name);
    lbi_vec->emplace_back(sub_lbi);
  };
  OperatorConf op_conf;
  op_conf.set_scope_symbol_id(scope_symbol_id);
  op_conf.set_device_tag(*JUST(DeviceTag4DeviceType(parallel_desc.device_type())));
  if (sbp.has_broadcast_parallel()) {
    op_conf.set_name(kAutoLocalBlobNamePrefix + "-DistributeClone-" + NewUniqueId());
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
        << "only `S(0)' global blob is compatible to local blob";
    op_conf.set_name(kAutoLocalBlobNamePrefix + "-DistributeSplit-" + NewUniqueId());
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
    OF_UNIMPLEMENTED() << "`P' global blob is not compatible to local blob";
  }
  {
    const auto& producer_op_conf = JUST(Op4OpName(lbi.op_name()))->op_conf();
    CHECK_OR_RETURN(producer_op_conf.has_scope_symbol_id());
    const auto& scope = Singleton<symbol::Storage<Scope>>::Get()->Get(scope_symbol_id);
    const auto* job_desc = JUST(scope.job_desc());
    JUST(AddAndInferOp(op_conf, parallel_desc.parallel_conf(), job_desc, false));
  }
  return local_lbi;
}

Maybe<void> LazyJobBuildAndInferCtx::Complete() {
  CHECK_GT_OR_RETURN(job().net().op_size(), 0)
      << " Sorry, nn.Graph need at least 1 op in net, but get 0 now.";
  auto compile_tc = std::make_unique<CostCounter<std::chrono::seconds>>(true, true);
  CHECK_NOTNULL(Singleton<JobDesc>::Get());
  // A global variable to get graph configurations.
  auto current_graph_config = std::make_unique<GlobalJobDescScope>(mut_job()->job_conf(), job_id());
  JobPassCtx job_pass_ctx(GlobalJobDesc());
  const auto job_name = job().job_conf().job_name();
  auto LogJob = [&](const std::string& name_suffix) -> void {
    std::string full_log_name =
        job_name + "-job_id_" + std::to_string(job_id()) + "-" + name_suffix;
    TeePersistentLogStream::Create(full_log_name)->Write(job());
    Singleton<OpGraph>::New(job());
    Singleton<OpGraph>::Get()->ToDotWithFilePath(full_log_name + ".dot");
    Singleton<OpGraph>::Delete();
  };
  std::string debug_pass_name = GetStringFromEnv("ONEFLOW_DEBUG_PASS", "");
  auto NeedLogJob = [&](const std::string& pass_name) -> bool {
    if ("ALL" == debug_pass_name) {
      return true;
    } else if (pass_name == debug_pass_name) {
      return true;
    } else {
      return false;
    }
  };
  int32_t pass_cnt = 0;
  const int64_t prev_v = FLAGS_v;
  auto DoPass = [&](const std::string& pass_name, int32_t cnt = 0) -> Maybe<void> {
    auto pass_tc = std::make_unique<CostCounter<std::chrono::milliseconds>>(true, true);
    VLOG(1) << job_name << " start compiling with pass"
            << " pass_cnt_" + std::to_string(pass_cnt) + "-" + pass_name
            << (cnt > 0 ? std::to_string(cnt) : "");
    if (unlikely(NeedLogJob(pass_name))) {
      std::string cnt_str = cnt > 0 ? std::to_string(cnt) : "";
      LogJob("pass_cnt_" + std::to_string(pass_cnt) + "-" + pass_name + cnt_str + "-before");
      FLAGS_v = 3;
    }
    JUST(JobPass4Name(pass_name)(mut_job(), &job_pass_ctx));
    if (unlikely(NeedLogJob(pass_name))) {
      FLAGS_v = prev_v;
      std::string cnt_str = cnt > 0 ? std::to_string(cnt) : "";
      LogJob("pass_cnt_" + std::to_string(pass_cnt) + "-" + pass_name + cnt_str + "-after");
    }
    VLOG(1) << job_name << " finish compiling with pass"
            << " pass_cnt_" + std::to_string(pass_cnt) + "-" + pass_name
            << (cnt > 0 ? std::to_string(cnt) : "");
    pass_tc->Count("[GraphCompile]" + job_name + " " + pass_name, 1, true);
    ++pass_cnt;
    return Maybe<void>::Ok();
  };

  if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    TeePersistentLogStream::Create(StrCat("forward_graph", job_id()))->Write(job());
    Singleton<OpGraph>::New(job());
    Singleton<OpGraph>::Get()->ToDotWithFilePath("forward_dlnet_" + std::to_string(job_id())
                                                 + "_op_graph.dot");
    Singleton<OpGraph>::Delete();
  }

  if (GlobalJobDesc().Bool("__is_user_function__")) {
    // insert pinned identity to prevent the loss, loss initial gradient and
    // variable gradient from being eliminated by IRRoundTripBeforeAD pass
    JUST(DoPass("InsertPinnedIdentityOpPass"));
    // prune the dangling constant which are the 0 gradients initialized by
    // the autograd engine for those tensors that have no gradients
    JUST(DoPass("EliminateDeadNodesPass"));
    JUST(DoPass("NormalizationExponentialAverageAutoTickPass"));
    JUST(DoPass("AutoMixedPrecision"));
    // prune depend OP and and add ctrl_in_op to op_conf accordingly
    // to express the same semantics and avoid performance loss
    JUST(DoPass("PruneDependOpPass"));
    JUST(DoPass("PruneAmpWhiteIdentityOpPass"));
    JUST(DoPass("OptimizerPlacementOptimizationPass"));
    // run FuseAddToOutputPass before IRRoundTripBeforeAD since add_2 maybe
    // fused as add_n in IRRoundTripBeforeAD pass
    JUST(DoPass("FuseAddToOutputPass"));
#ifdef WITH_MLIR
    JUST(DoPass("IRRoundTripBeforeAD"));
#endif  // WITH_MLIR
    // run DynamicLossScaleSchedulePass, AutoTrainStep and AutoLearningRate
    // after IRRoundTripBeforeAD since IRRoundTripBeforeAD will do DCE
    // optimization which could eliminate the nodes inserted by them
    JUST(DoPass("DynamicLossScaleSchedulePass"));
    JUST(DoPass("AutoTrainStep"));
    JUST(DoPass("AutoLearningRate"));
    JUST(DoPass("QuantAwareTraining"));
    JUST(DoPass("GenerateOptimizerOpConfs"));
    // pinned identity can be pruned since GenerateOptimizerOpConfs pass has
    // already construct a complete computational graph
    JUST(DoPass("PrunePinnedIdentityOpPass"));
    JUST(DoPass("ReplaceEmbeddingOps"));
    JUST(DoPass("SequentialOneEmbeddingOpsPass"));
    JUST(DoPass("FuseEmbeddingShuffleInteractionPass"));
    JUST(DoPass("FuseBCEReduceMeanFwBwPass"));
    JUST(DoPass("AddSspVariableProxy"));
    JUST(DoPass("CheckpointingPass"));
    JUST(DoPass("CudnnFusedNormalizationAddReluPass"));
    JUST(DoPass("PruneCastToStaticShapeOpsPass"));
#ifdef WITH_MLIR
    JUST(DoPass("IRRoundTrip"));
#endif  // WITH_MLIR
    // run this pass again to fuse ops created in the first run.
    // TODO(guoran): loop multiple times inside the pass
    JUST(DoPass("FuseAddToOutputPass", 1));
    JUST(DoPass("FuseConsecutiveAddPass"));
    JUST(DoPass("IndexedSlicesOptimizerRewritePass"));
    JUST(DoPass("SplitSparseSoftmaxCrossEntropyOpPass"));
    JUST(DoPass("DoParallelCastBeforeWideningTypeCast"));
    JUST(DoPass("FuseCastScalePass"));
    JUST(DoPass("PruneParallelCastOpsPass"));
    JUST(DoPass("FuseUpdateOpsPass"));
    JUST(DoPass("FuseModelUpdateCastOpsPass"));
    JUST(DoPass("MultiTensorModelUpdatePass"));
    JUST(DoPass("FixPipelineStageIdPass"));
    JUST(DoPass("PipelineBufferPass"));
    JUST(DoPass("AutoParallelPass"));
    JUST(DoPass("DelayVariableOpExecutionPass"));
#ifdef WITH_CUTLASS
    JUST(DoPass("CutlassConvTuningWarmupPass"));
#endif  // WITH_CUTLASS
    JUST(DoPass("DumpVariableInfoPass"));
  }
  JUST(DoPass("DumpBlobParallelConfPass"));
  JUST(CheckJob());
  compile_tc->Count("[GraphCompile]" + job_name + " OptimizationLogicalGraph", 0);
  return Maybe<void>::Ok();
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
  std::string nomarl_array[] = {"at_int32",  "at_int64", "at_bool",   "at_float",    "at_double",
                                "at_string", "at_shape", "at_stride", "at_data_type"};
  std::string list_array[] = {"at_list_int32",     "at_list_int64", "at_list_float",
                              "at_list_data_type", "at_list_shape", "at_list_stride",
                              "at_list_string"};
  nlohmann::json attr_json = user_conf["attr"];
  for (int32_t i = 0; i < attr_json.size(); i++) {
    std::string key = attr_json[i]["key"];
    nlohmann::json value_json = attr_json[i]["value"];
    bool is_found_normal = false;
    for (int32_t j = 0; j < nomarl_array->length(); j++) {
      std::string value_key = nomarl_array[j];
      if (value_json.contains(value_key)) {
        is_found_normal = true;
        if ("at_shape" == value_key || "at_stride" == value_key) {
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
  HashSet<std::string> inputs_op_names;
  HashSet<std::string> outputs_op_names;
  std::vector<nlohmann::json> layers_vec;
  layers_vec.reserve(op_name2op_.size());
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
      inputs_op_names.insert(op_name);
    }
    if (op->op_conf().has_output_conf() && op->op_conf().has_return_conf()) {
      outputs_op_names.insert(op_name);
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
    inbound_nodes_vec.reserve(inbound_nodes.size());
    for (const auto& in_node_name : inbound_nodes) { inbound_nodes_vec.emplace_back(in_node_name); }
    json_layers_pair["inbound_nodes"] = inbound_nodes_vec;

    layers_vec.emplace_back(json_layers_pair);
  }

  nlohmann::json json_pair;
  json_pair["name"] = job_name;
  json_pair["layers"] = layers_vec;
  json_pair["input_layers"] = inputs_op_names;
  json_pair["output_layers"] = outputs_op_names;

  return json_pair.dump();
}

Maybe<void> JobBuildAndInferCtx::Rebuild() {
  // clear old state
  lbi2logical_blob_desc_.clear();
  lbi2nd_sbp_from_producer_view_.clear();
  lbi2parallel_desc_from_producer_view_.clear();
  lbi2disable_boxing_.clear();
  op_name2op_.clear();
  parallel_desc2placement_group_.clear();
  parallel_desc2blob_placement_group_.clear();
  global_lbi2local_lbi_.clear();
  local_lbi2sub_lbis_.clear();
  local_lbi2parallel_desc_.clear();
  local_lbi2sbp_parallel_.clear();
  op_name2ancestors_need_no_grad_.clear();
  // record op mirror view
  HashMap<std::string, bool> op_name2is_local;
  CHECK_OR_RETURN(job_->has_job_parallel_view_conf());
  for (const auto& op_conf : job_->net().op()) {
    const auto& op_name = op_conf.name();
    CHECK_OR_RETURN(op_name2is_local.find(op_name) == op_name2is_local.end());  // NOLINT
    op_name2is_local[op_name] = false;
    const auto& op_name2is_local_parallel_view =
        job_->job_parallel_view_conf().op_name2is_local_parallel_view();
    if (op_name2is_local_parallel_view.find(op_name) != op_name2is_local_parallel_view.end()) {
      if (op_name2is_local_parallel_view.at(op_name)) { op_name2is_local[op_name] = true; }
    }
  }
  // build op graph
  OpGraph op_graph;
  if (Singleton<JobDesc>::Get()) {
    JUST(op_graph.Init(*job_));
  } else {
    auto scope = std::make_unique<GlobalJobDescScope>(job_->job_conf(), job_id());
    JUST(op_graph.Init(*job_));
  }
  // clear old job except job_conf
  job_->mutable_net()->Clear();
  job_->mutable_placement()->Clear();
  job_->mutable_job_parallel_view_conf()->Clear();
  job_->mutable_helper()->Clear();
  // topo traverse op_graph to AddAndInferOp
  op_graph.TopoForEachNode([&](OpNode* node) -> void {
    const auto& op_conf = node->op().op_conf();
    CHECK(op_name2is_local.find(op_conf.name()) != op_name2is_local.end());
    bool is_local = op_name2is_local.at(op_conf.name());
    if (is_local) {
      CHECK_JUST(AddAndInferLocalOp(op_conf));
    } else {
      CHECK_JUST(AddAndInferGlobalOp(op_conf));
    }
  });
  // updata job_helper
  op_graph.DumpLogicalBlobDesc(job_);
  op_graph.DumpNdSbpSignature(job_);
  return Maybe<void>::Ok();
}

Maybe<std::string> JobBuildAndInferCtx::GetOpBlobLbn(const std::string& op_name,
                                                     const std::string& bn_in_op) const {
  const auto& lbi = JUST(Op4OpName(op_name))->BnInOp2Lbi(bn_in_op);
  return GenLogicalBlobName(lbi);
}

Maybe<std::string> JobBuildAndInferCtx::NewUniqueOpNameByFunctionalOpConf(
    const OperatorConf& op_conf) {
  // NOTE(chengcheng): arg op_conf has a default global op_name because it is created by
  //  static functional op expr, so we need reset a unique op name for each functional op.
  //  This op_conf can NOT be a input/output/variable op which has set correct name in nn.Graph.
  //  But free eager tensor is treated as a special variable which needs to create name here.
  CHECK_OR_RETURN(!(op_conf.has_input_conf() || op_conf.has_output_conf()));

  const auto& scope = JUST(GetCurrentScope());

  std::string op_name_prefix;
  for (const std::string& prefix : scope->scope_proto().scope_op_name_prefixes()) {
    op_name_prefix += (prefix + "-");
  }
  std::string op_type_name;
  if (op_conf.has_user_conf()) {
    op_type_name = op_conf.user_conf().op_type_name();
  } else if (op_conf.has_variable_conf()) {
    // NOTE(chengcheng): To support Free Eager Tensor caught by nn.Graph
    op_type_name = "FreeEagerTensor";
  } else {
    op_type_name = "SystemOp";
  }
  std::string op_name = op_name_prefix + op_type_name + "-" + std::to_string(unique_op_name_index_);
  ++unique_op_name_index_;

  return op_name;
}

}  // namespace oneflow
