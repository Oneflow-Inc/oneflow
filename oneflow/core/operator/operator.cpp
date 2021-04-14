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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/job/mirrored_sig_infer_hint.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/job/parallel_distribution_infer_hint.h"
#include "oneflow/core/job/foreign_callback.h"

namespace oneflow {

namespace {

DataType GetDataTypeFromBnInOpVec(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const PbRpf<std::string>& bn_in_ops) {
  for (const std::string& bn_in_op : bn_in_ops) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn_in_op);
    if (blob_desc) { return blob_desc->data_type(); }
  }
  return DataType::kInvalidDataType;
}

std::shared_ptr<Operator> CheckAndConstructOp(std::shared_ptr<const OperatorConf> op_conf) {
  Operator* rptr = NewObj<int32_t, Operator>(op_conf->op_type_case(), *op_conf);
  DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(op_conf->device_tag()));
  if (IsCpuOnly(*op_conf)) { CHECK_EQ(device_type, DeviceType::kCPU); }
  rptr->Init(op_conf);
  return std::shared_ptr<Operator>(rptr);
}

}  // namespace

Operator::Operator() : device_type_(DeviceType::kInvalidDevice) {}

void Operator::Init(const OperatorConf& op_conf) {
  Init(std::make_shared<const OperatorConf>(op_conf));
}

void Operator::Init(std::shared_ptr<const OperatorConf> op_conf) {
  op_conf_ = std::move(op_conf);
  device_type_ = CHECK_JUST(DeviceType4DeviceTag(op_conf_->device_tag()));
  InitFromOpConf();
  input_output_bns_.Reserve(input_bns().size() + output_bns().size());
  for (const auto& bn : input_bns()) { *input_output_bns_.Add() = bn; }
  for (const auto& bn : output_bns()) { *input_output_bns_.Add() = bn; }
}

const LogicalBlobId& Operator::BnInOp2Lbi(const std::string& bn_in_op) const {
  return arg_signature_.bn_in_op2lbi().at(bn_in_op);
}

const OperatorConf& Operator::op_conf() const {
  CHECK(op_conf_);
  return *op_conf_;
}

std::shared_ptr<const OperatorConf> Operator::shared_op_conf() const { return op_conf_; }

DeviceType Operator::device_type() const { return device_type_; }

const std::string& Operator::SoleIbn() const {
  CHECK_EQ(input_bns().size(), 1);
  return input_bns().Get(0);
}
const std::string& Operator::SoleObn() const {
  CHECK_EQ(output_bns().size(), 1);
  return output_bns().Get(0);
}
const std::string& Operator::SoleTbn() const {
  CHECK_EQ(tmp_bns().size(), 1);
  return tmp_bns().Get(0);
}

Maybe<const std::string*> Operator::obn4lbi(const LogicalBlobId& lbi) const {
  const auto& it = lbi2output_index_.find(lbi);
  CHECK_OR_RETURN(it != lbi2output_index_.end())
      << "no logical blob id found. lbn: " << lbi.op_name() << "/" << lbi.blob_name();
  return &output_bns().Get(it->second);
}

const PbRpf<std::string>& Operator::input_bns() const { return input_bns_; }

const PbRpf<std::string>& Operator::output_bns() const { return output_bns_; }

const PbRpf<std::string>& Operator::tmp_bns() const { return tmp_bns_; }

const PbRpf<std::string>& Operator::input_output_bns() const { return input_output_bns_; }

Maybe<void> Operator::InferParallelSignatureIf() {
  JUST(InferBlobParallelDesc());
  return Maybe<void>::Ok();
}

Maybe<const ParallelDesc> Operator::GetParallelDesc4BnInOp(const std::string& bn) const {
  CHECK_OR_RETURN(bn2parallel_desc_);
  auto it = bn2parallel_desc_->find(bn);
  CHECK_OR_RETURN(it != bn2parallel_desc_->end());
  return it->second;
}

Maybe<void> Operator::FillBlobParallelDesc(
    const std::function<Maybe<const ParallelDesc>(const std::string&)>& ParallelDesc4Bn) {
  CHECK_OR_RETURN(!bn2parallel_desc_);
  bn2parallel_desc_.reset(new HashMap<std::string, std::shared_ptr<const ParallelDesc>>);
  for (const auto& bn : input_output_bns()) {
    auto blob_parallel_desc = JUST(ParallelDesc4Bn(bn));
    CHECK(bn2parallel_desc_->emplace(bn, blob_parallel_desc).second);
  }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferBlobParallelDesc() {
  JUST(FillBlobParallelDesc(
      [&](const std::string& bn) -> Maybe<const ParallelDesc> { return GetOpParallelDesc(); }));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::FillOpParallelDesc(const ParallelDesc& parallel_desc) {
  return FillOpParallelDesc(std::make_shared<const ParallelDesc>(parallel_desc));
}

Maybe<void> Operator::FillOpParallelDesc(std::shared_ptr<const ParallelDesc> parallel_desc) {
  CHECK_OR_RETURN(!op_parallel_desc_);
  op_parallel_desc_ = std::move(parallel_desc);
  return Maybe<void>::Ok();
}

Maybe<const ParallelDesc> Operator::GetOpParallelDesc() const {
  CHECK_OR_RETURN(op_parallel_desc_);
  return op_parallel_desc_;
}

namespace {

Maybe<void> FillLogicalBlobDesc(
    const std::function<Maybe<const BlobDesc>(int32_t)>& BlobDesc4Index,
    const PbRpf<std::string>& bns,
    std::unique_ptr<std::vector<std::shared_ptr<const BlobDesc>>>* index2logical_blob_desc_ptr) {
  CHECK_OR_RETURN(!(*index2logical_blob_desc_ptr));
  index2logical_blob_desc_ptr->reset(new std::vector<std::shared_ptr<const BlobDesc>>());
  (*index2logical_blob_desc_ptr)->reserve(bns.size());
  for (int32_t i = 0; i < bns.size(); ++i) {
    (*index2logical_blob_desc_ptr)->emplace_back(JUST(BlobDesc4Index(i)));
  }
  return Maybe<void>::Ok();
}

Maybe<void> FillLogicalBlobDesc(
    const std::function<const BlobDesc&(const std::string&)>& BlobDesc4BnInOp,
    const PbRpf<std::string>& bns,
    std::unique_ptr<std::vector<std::shared_ptr<const BlobDesc>>>* index2logical_blob_desc_ptr) {
  CHECK_OR_RETURN(!(*index2logical_blob_desc_ptr));
  index2logical_blob_desc_ptr->reset(new std::vector<std::shared_ptr<const BlobDesc>>());
  (*index2logical_blob_desc_ptr)->reserve(bns.size());
  for (const auto& bn : bns) {
    const BlobDesc& blob_desc = BlobDesc4BnInOp(bn);
    (*index2logical_blob_desc_ptr)->emplace_back(std::make_shared<const BlobDesc>(blob_desc));
  }
  return Maybe<void>::Ok();
}

Maybe<void> FillLogicalBlobDesc(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const PbRpf<std::string>& bns,
    std::unique_ptr<std::vector<std::shared_ptr<const BlobDesc>>>* index2logical_blob_desc_ptr) {
  JUST(FillLogicalBlobDesc(
      [&](const std::string& bn) -> const BlobDesc& {
        const BlobDesc* blob_desc = BlobDesc4BnInOp(bn);
        CHECK_NOTNULL(blob_desc);
        return *blob_desc;
      },
      bns, index2logical_blob_desc_ptr));
  return Maybe<void>::Ok();
}

Maybe<const BlobDesc> GetLogicalBlobDesc(
    const std::unique_ptr<std::vector<std::shared_ptr<const BlobDesc>>>& index2logical_blob_desc,
    int32_t index) {
  CHECK_OR_RETURN(index2logical_blob_desc);
  CHECK_LT_OR_RETURN(index, index2logical_blob_desc->size());
  return index2logical_blob_desc->at(index);
}

Maybe<void> FillLogicalBlobDescSignature(
    const PbRpf<std::string>& bns,
    const std::unique_ptr<std::vector<std::shared_ptr<const BlobDesc>>>& index2logical_blob_desc,
    PbMap<std::string, BlobDescProto>* bn_in_op2blob_desc) {
  CHECK_OR_RETURN(index2logical_blob_desc);
  CHECK_EQ_OR_RETURN(bns.size(), index2logical_blob_desc->size());
  for (int32_t i = 0; i < bns.size(); ++i) {
    index2logical_blob_desc->at(i)->ToProto(&(*bn_in_op2blob_desc)[bns.Get(i)]);
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> Operator::FillLogicalInBlobDesc(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  JUST(FillLogicalBlobDesc(BlobDesc4BnInOp, input_bns(), &input_index2logical_blob_desc_));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::FillLogicalInBlobDesc(
    const std::function<const BlobDesc&(const std::string&)>& BlobDesc4BnInOp) {
  JUST(FillLogicalBlobDesc(BlobDesc4BnInOp, input_bns(), &input_index2logical_blob_desc_));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::FillLogicalInBlobDesc(
    const std::function<Maybe<const BlobDesc>(int32_t)>& BlobDesc4InputIndex) {
  JUST(FillLogicalBlobDesc(BlobDesc4InputIndex, input_bns(), &input_index2logical_blob_desc_));
  return Maybe<void>::Ok();
}

Maybe<const BlobDesc> Operator::GetLogicalBlobDesc4Ibn(const std::string& ibn) const {
  return GetLogicalBlobDesc4InputIndex(JUST(GetInputIndex(ibn)));
}

Maybe<const BlobDesc> Operator::GetLogicalBlobDesc4InputIndex(int32_t index) const {
  return GetLogicalBlobDesc(input_index2logical_blob_desc_, index);
}

Maybe<void> Operator::FillLogicalOutBlobDesc(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  JUST(FillLogicalBlobDesc(BlobDesc4BnInOp, output_bns(), &output_index2logical_blob_desc_));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::FillLogicalOutBlobDesc(
    const std::function<const BlobDesc&(const std::string&)>& BlobDesc4BnInOp) {
  JUST(FillLogicalBlobDesc(BlobDesc4BnInOp, output_bns(), &output_index2logical_blob_desc_));
  return Maybe<void>::Ok();
}

Maybe<const BlobDesc> Operator::GetLogicalBlobDesc4Obn(const std::string& obn) const {
  return GetLogicalBlobDesc4OutputIndex(JUST(GetOutputIndex(obn)));
}

Maybe<const BlobDesc> Operator::GetLogicalBlobDesc4OutputIndex(int32_t index) const {
  return GetLogicalBlobDesc(output_index2logical_blob_desc_, index);
}

Maybe<const BlobDesc*> Operator::GetLogicalBlobDescPtr4OutputIndex(int32_t index) const {
  CHECK_OR_RETURN(output_index2logical_blob_desc_);
  CHECK_LT_OR_RETURN(index, output_index2logical_blob_desc_->size());
  CHECK_OR_RETURN(output_index2logical_blob_desc_->at(index));
  return output_index2logical_blob_desc_->at(index).get();
}

Maybe<const BlobDesc> Operator::GetLogicalBlobDesc4BnInOp(const std::string& bn) const {
  const auto& it = bn2index_pair_.find(bn);
  CHECK_OR_RETURN(it != bn2index_pair_.end());
  if (it->second.first == BlobNameTag::kInputBlobName) {
    return GetLogicalBlobDesc4InputIndex(it->second.second);
  } else if (it->second.first == BlobNameTag::kOutputBlobName) {
    return GetLogicalBlobDesc4OutputIndex(it->second.second);
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
}

Maybe<void> Operator::InferLogicalOutBlobDescsIf() {
  CHECK_OR_RETURN(input_index2logical_blob_desc_);
  CHECK_OR_RETURN(!output_index2logical_blob_desc_);
  std::vector<std::shared_ptr<BlobDesc>> output_logical_blob_desc_vec;
  output_logical_blob_desc_vec.resize(output_bns().size());
  for (auto& blob_desc : output_logical_blob_desc_vec) {
    blob_desc.reset(new BlobDesc(DataType::kInvalidDataType));
  }
  std::vector<std::shared_ptr<BlobDesc>> in_logical_blob_desc_vec;
  in_logical_blob_desc_vec.resize(input_bns().size());
  auto BlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
    const auto& it = bn2index_pair_.find(bn);
    CHECK(it != bn2index_pair_.end());
    if (it->second.first == BlobNameTag::kInputBlobName) {
      auto& ptr = in_logical_blob_desc_vec.at(it->second.second);
      if (!ptr) { ptr.reset(new BlobDesc(*input_index2logical_blob_desc_->at(it->second.second))); }
      return ptr.get();
    } else if (it->second.first == BlobNameTag::kOutputBlobName) {
      return output_logical_blob_desc_vec.at(it->second.second).get();
    } else {
      UNIMPLEMENTED();
    }
  };
  JUST(InferLogicalOutBlobDescs(BlobDesc4BnInOp, *JUST(GetOpParallelDesc())));
  output_index2logical_blob_desc_.reset(new std::vector<std::shared_ptr<const BlobDesc>>());
  output_index2logical_blob_desc_->resize(output_bns().size());
  for (int32_t i = 0; i < output_bns().size(); ++i) {
    output_index2logical_blob_desc_->at(i) = output_logical_blob_desc_vec.at(i);
  }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferBlobDescsIf(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const JobDesc* job_desc) const {
  JUST(InferOutBlobDescsIf(GetBlobDesc4BnInOp, parallel_ctx));
  JUST(InferInternalBlobDescsIf(GetBlobDesc4BnInOp, parallel_ctx, job_desc));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferOutBlobDescsIf(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  return InferOutBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

Maybe<void> Operator::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  if (parallel_ctx->parallel_num() == 1) {
    JUST(InferLogicalOutBlobDescs(GetBlobDesc4BnInOp, *JUST(GetOpParallelDesc())));
  } else {
    const auto& parallel_distribution_signature = JUST(this->parallel_distribution_signature());
    const auto& parallel_desc = JUST(this->GetOpParallelDesc());
    for (const auto& bn : input_bns()) {
      const auto& parallel_distribution =
          parallel_distribution_signature->bn_in_op2parallel_distribution().at(bn);
      std::shared_ptr<const BlobDesc> in_logical = JUST(GetLogicalBlobDesc4Ibn(bn));
      CHECK_OR_RETURN(*JUST(GetPhysicalShape(in_logical->shape(), parallel_distribution,
                                             *parallel_desc, *parallel_ctx))
                      == GetBlobDesc4BnInOp(bn)->shape());
    }
    for (const auto& bn : output_bns()) {
      BlobDesc* desc = GetBlobDesc4BnInOp(bn);
      *desc = *JUST(GetLogicalBlobDesc4Obn(bn));
      const auto& parallel_distribution =
          parallel_distribution_signature->bn_in_op2parallel_distribution().at(bn);
      desc->mut_shape() = *JUST(
          GetPhysicalShape(desc->shape(), parallel_distribution, *parallel_desc, *parallel_ctx));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferInternalBlobDescsIf(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const JobDesc* job_desc) const {
  return InferInternalBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, job_desc);
}

Maybe<void> Operator::InferInternalBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const JobDesc* job_desc) const {
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferInplaceObn2IbnIf(
    HashMap<std::string, std::string>* mut_inplace_obn2ibn,
    HashMap<std::string, std::string>* con_inplace_obn2ibn,
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  return InferInplaceObn2Ibn(mut_inplace_obn2ibn, con_inplace_obn2ibn, GetBlobDesc4BnInOp,
                             parallel_ctx);
}

Maybe<void> Operator::InferInplaceObn2Ibn(
    HashMap<std::string, std::string>* mut_inplace_obn2ibn,
    HashMap<std::string, std::string>* con_inplace_obn2ibn,
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  for (const std::string& obn : output_bns()) {
    const auto& obn_modifier = OutputBlobModifier4Obn(obn);
    if (obn_modifier.has_mutable_inplace_ibn()) {
      mut_inplace_obn2ibn->emplace(obn, obn_modifier.mutable_inplace_ibn());
    } else if (obn_modifier.has_const_inplace_ibn()) {
      con_inplace_obn2ibn->emplace(obn, obn_modifier.const_inplace_ibn());
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::FillInputBlobTimeShape(
    const std::function<Maybe<const Shape>(int32_t)>& GetTimeShape4InputIndex) {
  CHECK_OR_RETURN(!input_index2time_shape_);
  input_index2time_shape_.reset(new std::vector<std::shared_ptr<const Shape>>());
  input_index2time_shape_->reserve(input_bns().size());
  for (int32_t i = 0; i < input_bns().size(); ++i) {
    std::shared_ptr<const Shape> time_shape = JUST(GetTimeShape4InputIndex(i));
    if ((!input_blob_fastest_time_shape_)
        || input_blob_fastest_time_shape_->elem_cnt() < time_shape->elem_cnt()) {
      input_blob_fastest_time_shape_ = time_shape;
    }
    input_index2time_shape_->emplace_back(time_shape);
  }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferOpTimeShapeIf() {
  CHECK_OR_RETURN(!op_time_shape_);
  CHECK_OR_RETURN(input_index2time_shape_);
  auto GetTimeShape4BnInOp = [&](const std::string& ibn) -> Maybe<const Shape> {
    const auto& it = bn2index_pair_.find(ibn);
    CHECK_OR_RETURN(it != bn2index_pair_.end());
    CHECK_EQ_OR_RETURN(it->second.first, kInputBlobName);
    return input_index2time_shape_->at(it->second.second);
  };
  JUST(InferOpTimeShape(GetTimeShape4BnInOp, &op_time_shape_));
  if (input_blob_fastest_time_shape_
      && input_blob_fastest_time_shape_->elem_cnt() > op_time_shape_->elem_cnt()) {
    input_output_fastest_time_shape_ = input_blob_fastest_time_shape_;
  } else {
    input_output_fastest_time_shape_ = op_time_shape_;
  }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferOpTimeShape(
    const std::function<Maybe<const Shape>(const std::string&)>& GetTimeShape4BnInOp,
    std::shared_ptr<const Shape>* time_shape) const {
  if (!input_bns().empty()) {
    std::shared_ptr<const Shape> first_time_shape = input_index2time_shape_->at(0);
    for (int64_t i = 1; i < input_bns().size(); ++i) {
      CHECK_EQ_OR_RETURN(*input_index2time_shape_->at(i), *first_time_shape);
    }
    *time_shape = first_time_shape;
  } else {
    *time_shape = std::make_shared<const Shape>(Shape({1, 1}));
  }
  return Maybe<void>::Ok();
}

Maybe<const Shape> Operator::GetOpTimeShape() const {
  CHECK_OR_RETURN(op_time_shape_);
  return op_time_shape_;
}

Maybe<const Shape> Operator::GetInputBlobFastestTimeShape() const {
  return input_blob_fastest_time_shape_;
}

Maybe<const Shape> Operator::GetInputOutputFastestTimeShape() const {
  return input_output_fastest_time_shape_;
}

Maybe<void> Operator::GetSbpSignaturesIf(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const {
  JUST(GetSbpSignatures(LogicalBlobDesc4Ibn, parallel_desc, sbp_sig_list));
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Broadcast(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

void Operator::ForEachBnInOp(std::function<void(const std::string&)> Handler) const {
  for (const std::string& bn_in_op : input_bns()) { Handler(bn_in_op); }
  for (const std::string& bn_in_op : output_bns()) { Handler(bn_in_op); }
  for (const std::string& bn_in_op : tmp_bns()) { Handler(bn_in_op); }
}

Maybe<void> Operator::FillSbpSignature(const SbpSignature& sbp_signature) {
  ParallelDistributionSignature parallel_distribution_signature;
  SbpSignatureToParallelDistributionSignature(sbp_signature, &parallel_distribution_signature);
  JUST(FillParallelDistributionSignature(parallel_distribution_signature));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::FillParallelDistributionSignature(
    const ParallelDistributionSignature& signature) {
  CHECK_OR_RETURN(!parallel_distribution_signature_);
  CHECK_OR_RETURN(!sbp_signature_);
  parallel_distribution_signature_.reset(new ParallelDistributionSignature(signature));
  CHECK_OR_RETURN(op_parallel_desc_);
  if (op_parallel_desc_->hierarchy()->NumAxes() == 1) {
    SbpSignature sbp_signature;
    ParallelDistributionSignatureToSbpSignature(signature, &sbp_signature);
    sbp_signature_.reset(new SbpSignature(sbp_signature));
  }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferSbpSignatureIf(
    const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) {
  SbpSignature signature;
  if (parallel_desc.parallel_num() == 1) {
    auto* bn2sbp = signature.mutable_bn_in_op2sbp_parallel();
    for (const auto& ibn : input_bns()) { (*bn2sbp)[ibn].mutable_split_parallel()->set_axis(0); }
    for (const auto& obn : output_bns()) { (*bn2sbp)[obn].mutable_split_parallel()->set_axis(0); }
  } else if (parallel_desc.parallel_num() > 1) {
    JUST(InferSbpSignature(&signature, sbp_sig_conf, CalcOrderValue4SbpSig, SbpInferHint4Ibn,
                           parallel_desc));
  } else {
    UNIMPLEMENTED();
  }
  JUST(FillSbpSignature(signature));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferSbpSignature(
    SbpSignature* infered_sbp_signature, const SbpSignature& sbp_sig_conf,
    const HashMap<std::string, SbpInferHint>& ibn2sbp_infer_hint) const {
  auto SbpInferHint4Ibn = [&](const std::string& ibn) -> Maybe<const SbpInferHint*> {
    auto it = ibn2sbp_infer_hint.find(ibn);
    if (it == ibn2sbp_infer_hint.end()) {
      return Error::CheckFailedError()
             << "cannot find corresponding SbpInferHint for input_blob_name : " << ibn;
    }
    return &(it->second);
  };
  std::function<int32_t(const SbpSignature&)> CalcOrderValue4SbpSig;
  auto OrderValue4SourceDefaultSplit0 = [&](const std::string& bn,
                                            const SbpParallel& sbp_parallel) -> int32_t {
    return -1 * (sbp_parallel.has_split_parallel() && sbp_parallel.split_parallel().axis() == 0);
  };
  auto OrderValue4SbpHint = [&](const std::string& ibn,
                                const SbpParallel& sbp_parallel) -> int32_t {
    const auto* hint = CHECK_JUST(SbpInferHint4Ibn(ibn));
    // NOTE(chengcheng): one to one connect.
    return -10
           * (hint->sbp_parallel() == sbp_parallel
              && hint->parallel_desc().parallel_num() == op_parallel_desc_->parallel_num());
  };

  if (sbp_sig_conf.bn_in_op2sbp_parallel().empty()) {
    CalcOrderValue4SbpSig = [&](const SbpSignature& sbp_signature) -> int32_t {
      int32_t order_value = 0;
      if (input_bns().size() > 0) {
        // NOTE(chengcheng): non-source op only ordered by input sbp match.
        for (const auto& ibn : input_bns()) {
          const auto& sbp_parallel_it = sbp_signature.bn_in_op2sbp_parallel().find(ibn);
          CHECK(sbp_parallel_it != sbp_signature.bn_in_op2sbp_parallel().end());
          order_value += OrderValue4SbpHint(ibn, sbp_parallel_it->second);
        }
      } else {
        // NOTE(chengcheng): source op default split(0)
        //   ONLY data source op will consider order here. variable op sbp is set by user.
        for (const auto& obn : output_bns()) {
          const auto& sbp_parallel_it = sbp_signature.bn_in_op2sbp_parallel().find(obn);
          CHECK(sbp_parallel_it != sbp_signature.bn_in_op2sbp_parallel().end());
          order_value += OrderValue4SourceDefaultSplit0(obn, sbp_parallel_it->second);
        }
      }
      return order_value;
    };
  } else {
    CalcOrderValue4SbpSig = [](const SbpSignature&) -> int32_t { return 0; };
  }
  if (op_parallel_desc_->parallel_num() == 1) {
    auto* bn2sbp = infered_sbp_signature->mutable_bn_in_op2sbp_parallel();
    for (const auto& ibn : input_bns()) { (*bn2sbp)[ibn].mutable_split_parallel()->set_axis(0); }
    for (const auto& obn : output_bns()) { (*bn2sbp)[obn].mutable_split_parallel()->set_axis(0); }
  } else if (op_parallel_desc_->parallel_num() > 1) {
    JUST(InferSbpSignature(infered_sbp_signature, sbp_sig_conf, CalcOrderValue4SbpSig,
                           SbpInferHint4Ibn, *op_parallel_desc_));
  } else {
    UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::FilterAndCheckValidSbpSignatureListByLogicalShape(
    const SbpSignatureList& total_sbp_sig_list,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc, SbpSignatureList* valid_sbp_sig_list) const {
  auto GetOpDebugShapeStr = [&]() -> std::string {
    std::string ret = "";
    if (op_conf().has_user_conf()) {
      ret += ("op_type_name = " + op_conf().user_conf().op_type_name() + ", ");
    }
    for (const auto& ibn : input_bns()) {
      ret +=
          (" ibn:(" + ibn + ") lbn:(" + GenLogicalBlobName(BnInOp2Lbi(ibn)) + ") logical_shape = "
           + CHECK_JUST(SbpInferHint4Ibn(ibn))->logical_blob_desc().shape().DebugStr() + ", ");
    }
    return ret;
  };
  for (const auto& sbp_signature : total_sbp_sig_list.sbp_signature()) {
    bool is_valid = true;

    for (const auto& ibn : input_bns()) {
      const auto& sbp_parallel_it = sbp_signature.bn_in_op2sbp_parallel().find(ibn);
      CHECK_OR_RETURN(sbp_parallel_it != sbp_signature.bn_in_op2sbp_parallel().end());
      const SbpParallel& sbp_parallel = sbp_parallel_it->second;
      const SbpInferHint* hint = JUST(SbpInferHint4Ibn(ibn));
      const Shape& logical_shape = hint->logical_blob_desc().shape();
      // NOTE(chengcheng): disable split when logical shape cannot split at this axis
      if (sbp_parallel.has_split_parallel()) {
        const int64_t axis = sbp_parallel.split_parallel().axis();
        CHECK_OR_RETURN(axis >= 0 && axis < logical_shape.NumAxes())
            << "The sbp sign is ERROR because of the split axis >= shape num axes. In op: ["
            << op_name() << "] ibn: (" << ibn << ") the split axis is = " << axis
            << " . And the logical_shape = " << logical_shape.DebugStr()
            << ". This Op debug str = {" << GetOpDebugShapeStr() << "}";
        if (logical_shape.At(axis) < parallel_desc.parallel_num()) {
          // NOTE(chengcheng): cannot split at this axis!
          is_valid = false;
          break;
        }
      }
    }
    if (is_valid) { *valid_sbp_sig_list->mutable_sbp_signature()->Add() = sbp_signature; }
  }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  // get op sbp signatures
  auto LogicalBlobDesc4Ibn = [&](const std::string& ibn) -> Maybe<const BlobDesc&> {
    const SbpInferHint* sbp_infer_hint = JUST(SbpInferHint4Ibn(ibn));
    return Maybe<const BlobDesc&>(sbp_infer_hint->logical_blob_desc());
  };
  SbpSignatureList valid_sbp_sig_list;
  {
    SbpSignatureList total_sbp_sig_list;
    JUST(GetSbpSignaturesIf(LogicalBlobDesc4Ibn, parallel_desc, &total_sbp_sig_list));
    // filter sbp signatures by logical shape
    JUST(FilterAndCheckValidSbpSignatureListByLogicalShape(total_sbp_sig_list, SbpInferHint4Ibn,
                                                           parallel_desc, &valid_sbp_sig_list));
  }
  // filter sbp signatures by sbp signature conf
  SbpSignatureList filtered_sbp_sigs_by_conf;
  FilterSbpSignatureList(valid_sbp_sig_list, sbp_sig_conf, &filtered_sbp_sigs_by_conf);
  CHECK_GT_OR_RETURN(filtered_sbp_sigs_by_conf.sbp_signature_size(), 0);
  if (filtered_sbp_sigs_by_conf.sbp_signature_size() == 1) {
    *sbp_signature = *filtered_sbp_sigs_by_conf.sbp_signature().begin();
    return Maybe<void>::Ok();
  }
  // sort sbp signatures by copy cost, then return the one with least cost
  HashMap<std::string, const SbpParallel*> ibn2producer_sbp_parallel;
  for (const auto& ibn : input_bns()) {
    ibn2producer_sbp_parallel[ibn] = &(JUST(SbpInferHint4Ibn(ibn))->sbp_parallel());
  }
  std::vector<const SbpSignature*> sorted_sbp_signatures;
  SortSbpSignatureListByCopyCost(filtered_sbp_sigs_by_conf, input_bns(), SbpInferHint4Ibn,
                                 CalcOrderValue4SbpSig, &sorted_sbp_signatures);
  *sbp_signature = *sorted_sbp_signatures.at(0);
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferParallelDistributionSignatureIf(
    const ParallelDistributionSignature& parallel_distribution_constraints,
    const ParallelDesc& parallel_desc,
    std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
        ParallelDistributionInferHint4Ibn) {
  ParallelDistributionSignature parallel_distribution_signature;
  JUST(InferParallelDistributionSignature(&parallel_distribution_signature,
                                          parallel_distribution_constraints, parallel_desc,
                                          ParallelDistributionInferHint4Ibn));
  JUST(FillParallelDistributionSignature(parallel_distribution_signature));
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferParallelDistributionSignature(
    ParallelDistributionSignature* parallel_distribution_signature,
    const ParallelDistributionSignature& parallel_distribution_constraints,
    const ParallelDesc& parallel_desc,
    std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
        ParallelDistributionInferHint4Ibn) const {
  const auto IsBroadcast = [](const ParallelDistribution& parallel_distribution,
                              const ParallelDesc& parallel_desc) -> bool {
    if (parallel_desc.parallel_num() == 1) { return true; }
    for (int64_t i = 0; i < parallel_distribution.sbp_parallel_size(); ++i) {
      if (!parallel_distribution.sbp_parallel(i).has_broadcast_parallel()) { return false; }
    }
    return true;
  };
  const auto& parallel_hierarchy = parallel_desc.hierarchy();
  CHECK_GT(parallel_hierarchy->NumAxes(), 0);
  if (parallel_hierarchy->NumAxes() == 1) {
    HashMap<std::string, SbpInferHint> ibn2sbp_infer_hint;
    for (const auto& ibn : input_bns()) {
      const ParallelDistributionInferHint* hint = JUST(ParallelDistributionInferHint4Ibn(ibn));
      if (hint->parallel_distribution().sbp_parallel_size() != 1) {
        CHECK_OR_RETURN(IsBroadcast(hint->parallel_distribution(), hint->parallel_desc()));
      }
      ibn2sbp_infer_hint.emplace(ibn,
                                 SbpInferHint(&hint->parallel_desc(), &hint->logical_blob_desc(),
                                              &hint->parallel_distribution().sbp_parallel(0)));
    }
    SbpSignature sbp_constraints;
    ParallelDistributionSignatureToSbpSignature(parallel_distribution_constraints,
                                                &sbp_constraints);
    SbpSignature sbp_signature;
    CHECK_JUST(InferSbpSignature(&sbp_signature, sbp_constraints, ibn2sbp_infer_hint));
    SbpSignatureToParallelDistributionSignature(sbp_signature, parallel_distribution_signature);
    return Maybe<void>::Ok();
  } else {
    SbpSignatureList list;
    const auto LogicalBlobDesc4Ibn = [&](const std::string& ibn) -> Maybe<const BlobDesc&> {
      return JUST(ParallelDistributionInferHint4Ibn(ibn))->logical_blob_desc();
    };
    HashMap<std::string, ParallelDistribution> ibn2parallel_distribution;
    for (const auto& ibn : input_bns()) {
      ParallelDistribution distribution =
          JUST(ParallelDistributionInferHint4Ibn(ibn))->parallel_distribution();
      if (parallel_distribution_constraints.bn_in_op2parallel_distribution_size() != 0) {
        const auto parallel_distribution_constraints_it =
            parallel_distribution_constraints.bn_in_op2parallel_distribution().find(ibn);
        if (parallel_distribution_constraints_it
            != parallel_distribution_constraints.bn_in_op2parallel_distribution().end()) {
          distribution = parallel_distribution_constraints_it->second;
        }
      }
      if (distribution.sbp_parallel_size() != parallel_hierarchy->NumAxes()) {
        CHECK_OR_RETURN(IsBroadcast(distribution,
                                    JUST(ParallelDistributionInferHint4Ibn(ibn))->parallel_desc()))
            << ibn << "'s hierarchy is different from " << op_name()
            << "'s, it should be broadcast but get " << distribution.DebugString();
        distribution.clear_sbp_parallel();
        for (int64_t i = 0; i < parallel_hierarchy->NumAxes(); ++i) {
          distribution.add_sbp_parallel()->mutable_broadcast_parallel();
        }
      }
      CHECK_EQ_OR_RETURN(distribution.sbp_parallel_size(), parallel_hierarchy->NumAxes());
      ibn2parallel_distribution[ibn] = std::move(distribution);
    }
    CHECK_JUST(GetSbpSignaturesIf(LogicalBlobDesc4Ibn, parallel_desc, &list));
    for (int64_t i = 0; i < parallel_hierarchy->NumAxes(); ++i) {
      const SbpSignature* matched_sbp_signature = nullptr;
      for (const auto& sbp_signature : list.sbp_signature()) {
        bool all_match = true;
        for (const auto& ibn : input_bns()) {
          if (sbp_signature.bn_in_op2sbp_parallel().at(ibn)
              != ibn2parallel_distribution.at(ibn).sbp_parallel(i)) {
            all_match = false;
            break;
          }
        }
        if (all_match) {
          matched_sbp_signature = &sbp_signature;
          break;
        }
      }
      CHECK_OR_RETURN(matched_sbp_signature != nullptr) << " op_name " << op_name();
      for (const auto& bn : input_bns()) {
        *((*parallel_distribution_signature->mutable_bn_in_op2parallel_distribution())[bn]
              .add_sbp_parallel()) = matched_sbp_signature->bn_in_op2sbp_parallel().at(bn);
      }
      for (const auto& bn : output_bns()) {
        *((*parallel_distribution_signature->mutable_bn_in_op2parallel_distribution())[bn]
              .add_sbp_parallel()) = matched_sbp_signature->bn_in_op2sbp_parallel().at(bn);
      }
    }
    return Maybe<void>::Ok();
  }
}

Maybe<void> Operator::InferMirroredSignatureIf(
    std::function<Maybe<const MirroredSigInferHint*>(const std::string&)> MirroredSigInferHint4Ibn,
    bool is_mirrored_parallel_view_conf, const ParallelDesc& parallel_desc) {
  return InferMirroredSignature(MirroredSigInferHint4Ibn, is_mirrored_parallel_view_conf,
                                parallel_desc);
}

std::string DebugString4MirroredHint(
    std::function<Maybe<const MirroredSigInferHint*>(const std::string&)> MirroredSigInferHint4Ibn,
    const Operator& op) {
  std::string ret;
  for (const auto& ibn : op.input_bns()) {
    const auto& infer_hint = *CHECK_JUST(MirroredSigInferHint4Ibn(ibn));
    bool is_mirrored = infer_hint.is_mirrored_parallel_view();
    ret += "arg: " + ibn + ", is_mirrored: " + (is_mirrored ? "true" : "false") + "\n";
  }
  return ret;
}

Maybe<void> Operator::InferMirroredSignature(
    std::function<Maybe<const MirroredSigInferHint*>(const std::string&)> MirroredSigInferHint4Ibn,
    bool is_mirrored_parallel_view_conf, const ParallelDesc& parallel_desc) {
  HashSet<bool> is_mirrored_parallel_view_values;
  for (const auto& ibn : input_bns()) {
    const auto& infer_hint = *JUST(MirroredSigInferHint4Ibn(ibn));
    is_mirrored_parallel_view_values.insert(infer_hint.is_mirrored_parallel_view());
  }
  CHECK_LE_OR_RETURN(is_mirrored_parallel_view_values.size(), 1)
      << "mixed parallel_views are disallowed."
      << "\n=========== is_mirrrored_conf ===========\n"
      << DebugString4MirroredHint(MirroredSigInferHint4Ibn, *this)
      << "\n=========== op_cnf ===========\n"
      << op_conf().DebugString();
  if (is_mirrored_parallel_view_values.size() == 1) {
    is_mirrored_parallel_view_conf = *is_mirrored_parallel_view_values.begin();
  }
  if (is_mirrored_parallel_view_conf) {
    for (const auto& ibn : input_bns()) {
      const auto& infer_hint = *JUST(MirroredSigInferHint4Ibn(ibn));
      CHECK_EQ_OR_RETURN(infer_hint.parallel_desc().parallel_num(), parallel_desc.parallel_num());
    }
  }
  const auto SetIsMirroredParallel = [&](const std::string& bn_in_op) {
    if (is_mirrored_parallel_view_conf) {
      MutOptMirroredParallel(bn_in_op)->mutable_mirrored_parallel();
    } else {
      MutOptMirroredParallel(bn_in_op)->clear_mirrored_parallel();
    }
  };
  for (const auto& ibn : input_bns()) { SetIsMirroredParallel(ibn); }
  for (const auto& obn : output_bns()) { SetIsMirroredParallel(obn); }
  return Maybe<void>::Ok();
}

Maybe<const SbpSignature*> Operator::sbp_signature() const {
  CHECK_OR_RETURN(sbp_signature_) << "sbp signature not infered";
  return sbp_signature_.get();
}

Maybe<const ParallelDistributionSignature*> Operator::parallel_distribution_signature() const {
  CHECK_OR_RETURN(parallel_distribution_signature_)
      << "parallel distribution signature not infered";
  return parallel_distribution_signature_.get();
}

BlobLastUsedSignature* Operator::mut_blob_last_used_signature() {
  if (!blob_last_used_signature_) { blob_last_used_signature_.reset(new BlobLastUsedSignature()); }
  return blob_last_used_signature_.get();
}

BlobBackwardUsedSignature* Operator::mut_blob_backward_used_signature() {
  if (!blob_backward_used_signature_) {
    blob_backward_used_signature_.reset(new BlobBackwardUsedSignature());
  }
  return blob_backward_used_signature_.get();
}

Maybe<const SbpParallel*> Operator::SbpParallel4BnInOp(const std::string& bn_in_op) const {
  CHECK_OR_RETURN(sbp_signature_) << "sbp signature not infered";
  const auto& map = sbp_signature_->bn_in_op2sbp_parallel();
  const auto& iter = map.find(bn_in_op);
  CHECK_OR_RETURN(iter != map.end()) << "blob_name " << bn_in_op << " not found in sbp signature";
  return &iter->second;
}

Maybe<const ParallelDistribution*> Operator::ParallelDistribution4BnInOp(
    const std::string& bn_in_op) const {
  CHECK_OR_RETURN(parallel_distribution_signature_)
      << "parallel distribution signature not infered";
  const auto& map = parallel_distribution_signature_->bn_in_op2parallel_distribution();
  const auto& iter = map.find(bn_in_op);
  CHECK_OR_RETURN(iter != map.end())
      << "blob_name " << bn_in_op << " not found in parallel distribution";
  return &iter->second;
}

Maybe<const OptMirroredParallel*> Operator::OptMirroredParallel4BnInOp(
    const std::string& bn_in_op) const {
  CHECK_OR_RETURN(mirrored_signature_) << "mirrored signature not infered";
  const auto& map = mirrored_signature_->bn_in_op2opt_mirrored_parallel();
  const auto& iter = map.find(bn_in_op);
  CHECK_OR_RETURN(iter != map.end())
      << "blob_name " << bn_in_op << " not found in mirrored signature";
  return &iter->second;
}

OptMirroredParallel* Operator::MutOptMirroredParallel(const std::string& bn_in_op) {
  if (!mirrored_signature_) { mirrored_signature_.reset(new MirroredSignature()); }
  auto* map = mirrored_signature_->mutable_bn_in_op2opt_mirrored_parallel();
  return &(*map)[bn_in_op];
}

namespace {

bool HasBlobDescWithField(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                          const PbRpf<std::string>& bn_in_ops,
                          std::function<bool(const BlobDesc*)> Predicator4BlobDesc) {
  for (const std::string& bn_in_op : bn_in_ops) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn_in_op);
    if (blob_desc && Predicator4BlobDesc(blob_desc)) { return true; }
  }
  return false;
}

}  // namespace

void Operator::GenKernelConf(
    const std::function<const BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  auto* dtype_signature = kernel_conf->mutable_dtype_signature();
  for (const std::string& ibn : input_bns()) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(ibn);
    if (blob_desc == nullptr) { continue; }
    (*dtype_signature->mutable_name2dtype())[ibn] = blob_desc->data_type();
  }
  CHECK_JUST(ToOpAttribute(kernel_conf->mutable_op_attribute()));
  if (HasBlobDescWithField(GetBlobDesc4BnInOp, output_bns(),
                           [](const BlobDesc* blob_desc) { return blob_desc->is_dynamic(); })) {
    kernel_conf->set_need_do_shape(true);
  }

  {
    DataType data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, output_bns());
    if (data_type == DataType::kInvalidDataType) {
      data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, input_bns());
    }
    kernel_conf->set_data_type(data_type);
  }
  if (parallel_ctx != nullptr) { *(kernel_conf->mutable_parallel_ctx()) = *parallel_ctx; }

  VirtualGenKernelConf(GetBlobDesc4BnInOp, parallel_ctx, kernel_conf);
}

void Operator::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {}

void Operator::AddLbi2OutputIndex(const LogicalBlobId& lbi, int32_t output_index) {
  CHECK(lbi2output_index_.emplace(lbi, output_index).second);
}

std::string Operator::Bn2ConfName(const std::string& bn) const {
  return GetStrValInPbFdOrPbRpf(GetCustomizedConf(), bn);
}

LogicalBlobId Operator::lbi4ibn(const std::string& input_bn) const {
  return GenLogicalBlobId(Bn2ConfName(input_bn));
}
LogicalBlobId Operator::lbi4obn(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(Bn2ConfName(output_bn));
  return ret;
}
LogicalBlobId Operator::tbn2lbi(const std::string& tmp_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(tmp_bn);
  return ret;
}

void Operator::EnrollTmpBn(const std::string& tbn) {
  *tmp_bns_.Add() = tbn;
  CHECK(mut_bn_in_op2lbi()->insert({tbn, tbn2lbi(tbn)}).second);
}

InputBlobModifier* Operator::EnrollInputBn(const std::string& ibn, bool has_diff) {
  LogicalBlobId lbi = lbi4ibn(ibn);
  auto* map = arg_modifier_signature_.mutable_ibn2input_blob_modifier();
  const auto& pair = map->insert({ibn, InputBlobModifier()});
  CHECK(pair.second);
  const int32_t input_index = input_bns_.size();
  CHECK(
      bn2index_pair_.emplace(ibn, std::make_pair(BlobNameTag::kInputBlobName, input_index)).second);
  *input_bns_.Add() = ibn;
  CHECK(mut_bn_in_op2lbi()->insert({ibn, lbi}).second);
  auto* ret = &pair.first->second;
  ret->set_requires_grad(has_diff);
  return ret;
}

const InputBlobModifier& Operator::InputBlobModifier4Ibn(const std::string& ibn) const {
  return arg_modifier_signature_.ibn2input_blob_modifier().at(ibn);
}

const OutputBlobModifier& Operator::OutputBlobModifier4Obn(const std::string& obn) const {
  return arg_modifier_signature_.obn2output_blob_modifier().at(obn);
}

InputBlobModifier* Operator::MutInputBlobModifier4Ibn(const std::string& ibn) {
  auto* map = arg_modifier_signature_.mutable_ibn2input_blob_modifier();
  return &map->at(ibn);
}

OutputBlobModifier* Operator::MutOutputBlobModifier4Obn(const std::string& obn) {
  auto* map = arg_modifier_signature_.mutable_obn2output_blob_modifier();
  return &map->at(obn);
}

void Operator::EnrollRepeatedInputBn(const std::string& ibn_prefix, int32_t num, bool has_diff) {
  FOR_RANGE(int32_t, i, 0, num) { EnrollInputBn(GenRepeatedBn(ibn_prefix, i), has_diff); }
}

void Operator::EnrollRepeatedInputBn(const std::string& ibn_prefix, bool has_diff) {
  EnrollRepeatedInputBn(ibn_prefix, GetPbRpfFromCustomizedConf<std::string>(ibn_prefix).size(),
                        has_diff);
}

void Operator::EnrollRepeatedInputBn(const std::string& ibn_prefix, int32_t num) {
  EnrollRepeatedInputBn(ibn_prefix, num, true);
}

void Operator::EnrollRepeatedInputBn(const std::string& ibn_prefix) {
  EnrollRepeatedInputBn(ibn_prefix, true);
}

OutputBlobModifier* Operator::EnrollOutputBn(const std::string& obn, bool has_diff) {
  LogicalBlobId lbi = lbi4obn(obn);
  auto* map = arg_modifier_signature_.mutable_obn2output_blob_modifier();
  const auto& pair = map->insert({obn, OutputBlobModifier()});
  CHECK(pair.second);
  auto* ret = &pair.first->second;
  const int32_t output_index = output_bns_.size();
  CHECK(bn2index_pair_.emplace(obn, std::make_pair(BlobNameTag::kOutputBlobName, output_index))
            .second);
  AddLbi2OutputIndex(lbi, output_index);
  *output_bns_.Add() = obn;
  CHECK(mut_bn_in_op2lbi()->insert({obn, lbi}).second);
  ret->set_requires_grad(has_diff);
  return ret;
}

void Operator::EnrollRepeatedOutputBnWithSetter(
    const std::string& obn_prefix, int32_t num, bool has_diff,
    const std::function<void(OutputBlobModifier*)>& ModifierSetter) {
  FOR_RANGE(int32_t, i, 0, num) {
    ModifierSetter(EnrollOutputBn(GenRepeatedBn(obn_prefix, i), has_diff));
  }
}

void Operator::EnrollRepeatedOutputBnWithSetter(
    const std::string& obn_prefix, bool has_diff,
    const std::function<void(OutputBlobModifier*)>& ModifierSetter) {
  EnrollRepeatedOutputBnWithSetter(obn_prefix,
                                   GetPbRpfFromCustomizedConf<std::string>(obn_prefix).size(),
                                   has_diff, ModifierSetter);
}

void Operator::EnrollRepeatedOutputBnWithSetter(
    const std::string& obn_prefix, int32_t num,
    const std::function<void(OutputBlobModifier*)>& ModifierSetter) {
  EnrollRepeatedOutputBnWithSetter(obn_prefix, num, true, ModifierSetter);
}

void Operator::EnrollRepeatedOutputBnWithSetter(
    const std::string& obn_prefix, const std::function<void(OutputBlobModifier*)>& ModifierSetter) {
  EnrollRepeatedOutputBnWithSetter(obn_prefix, true, ModifierSetter);
}

void Operator::EnrollRepeatedOutputBn(const std::string& obn_prefix, int32_t num, bool has_diff) {
  FOR_RANGE(int32_t, i, 0, num) { EnrollOutputBn(GenRepeatedBn(obn_prefix, i), has_diff); }
}

void Operator::EnrollRepeatedOutputBn(const std::string& obn_prefix, bool has_diff) {
  EnrollRepeatedOutputBn(obn_prefix, GetPbRpfFromCustomizedConf<std::string>(obn_prefix).size(),
                         has_diff);
}

void Operator::EnrollRepeatedOutputBn(const std::string& obn_prefix, int32_t num) {
  EnrollRepeatedOutputBn(obn_prefix, num, true);
}

void Operator::EnrollRepeatedOutputBn(const std::string& obn_prefix) {
  EnrollRepeatedOutputBn(obn_prefix, true);
}

std::string GenRepeatedBn(const std::string& bn_prefix, int32_t idx) {
  CHECK_GE(idx, 0);
  return bn_prefix + "_" + std::to_string(idx);
}

std::pair<std::string, int32_t> GenUnRepeatedBn(const std::string& bn) {
  return GetFieldNameAndIndex4StrVal(bn);
}

bool IsCpuOnly(const OperatorConf& op_conf) {
  OperatorConf::OpTypeCase op_type_case = op_conf.op_type_case();
  using CpuOnly = OnlyCpuSupportPredicator;
  auto* ptr = NewObj<int32_t, CpuOnly>(op_type_case);
  CHECK(ptr != nullptr) << "op_conf\n" << op_conf.DebugString();
  if (*std::unique_ptr<CpuOnly>(ptr)) { return true; }
  if (!op_conf.has_user_conf()) { return false; }
  auto* registration_val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_conf.user_conf().op_type_name());
  CHECK_NOTNULL(registration_val);
  return registration_val->cpu_only_supported;
}

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf, DeviceType device_type) {
  std::shared_ptr<OperatorConf> dev_op_conf = std::make_shared<OperatorConf>(op_conf);
  dev_op_conf->set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(device_type)));
  auto op = CheckAndConstructOp(dev_op_conf);
  return op;
}

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf) {
  if (IsCpuOnly(op_conf)) { return ConstructOp(op_conf, DeviceType::kCPU); }
  return CheckAndConstructOp(std::make_shared<OperatorConf>(op_conf));
}

Symbol<OperatorConf> Operator::GetOpConfWithoutOpNameAndLbn() const {
  OperatorConf op_conf(this->op_conf());
  op_conf.set_name("undefined-op-name");
  PbMessage* op_type_conf = MutableMessageInPbMessage(&op_conf, op_conf.op_type_case());
  for (const auto& ibn : input_bns()) {
    if (!HasStrFieldInPbFdOrPbRpf(*op_type_conf, ibn)) { continue; }
    ReplaceInputLbnInOpCustomizedConf(&op_conf, ibn, "undefined-op-name/undefined-ibn");
  }
  return SymbolOf(op_conf);
}

std::shared_ptr<OpAttribute> Operator::GetOpAttributeWithoutOpNameAndLbn() const {
  auto op_attribute = std::make_shared<OpAttribute>();
  CHECK_JUST(ToOpAttribute(op_attribute.get()));
  op_attribute->mutable_sbp_signature();
  *op_attribute->mutable_op_conf() = *GetOpConfWithoutOpNameAndLbn();
  return op_attribute;
}

Maybe<int32_t> Operator::GetInputIndex(const std::string& ibn) const {
  auto it = bn2index_pair_.find(ibn);
  CHECK_OR_RETURN(it != bn2index_pair_.end());
  CHECK_EQ_OR_RETURN(it->second.first, BlobNameTag::kInputBlobName);
  return it->second.second;
}

Maybe<int32_t> Operator::GetOutputIndex(const std::string& obn) const {
  auto it = bn2index_pair_.find(obn);
  CHECK_OR_RETURN(it != bn2index_pair_.end());
  CHECK_EQ_OR_RETURN(it->second.first, BlobNameTag::kOutputBlobName);
  return it->second.second;
}

Maybe<int32_t> Operator::GetOutputIndex(const LogicalBlobId& lbi) const {
  auto it = lbi2output_index_.find(lbi);
  CHECK_OR_RETURN(it != lbi2output_index_.end());
  return it->second;
}

Maybe<void> Operator::ToOpAttribute(OpAttribute* op_attribute) const {
  *op_attribute->mutable_input_bns() = input_bns_;
  *op_attribute->mutable_output_bns() = output_bns_;
  *op_attribute->mutable_tmp_bns() = tmp_bns_;
  *op_attribute->mutable_op_conf() = op_conf();
  *op_attribute->mutable_arg_signature() = arg_signature_;
  *op_attribute->mutable_arg_modifier_signature() = arg_modifier_signature_;
  if (blob_last_used_signature_) {
    *op_attribute->mutable_blob_last_used_signature() = *blob_last_used_signature_;
  } else {
    op_attribute->clear_blob_last_used_signature();
  }
  if (blob_backward_used_signature_) {
    *op_attribute->mutable_blob_backward_used_signature() = *blob_backward_used_signature_;
  } else {
    op_attribute->clear_blob_backward_used_signature();
  }
  if (sbp_signature_) {
    *op_attribute->mutable_sbp_signature() = *sbp_signature_;
  } else {
    op_attribute->clear_sbp_signature();
  }
  if (parallel_distribution_signature_) {
    *op_attribute->mutable_parallel_distribution_signature() = *parallel_distribution_signature_;
  } else {
    op_attribute->clear_parallel_distribution_signature();
  }
  if (mirrored_signature_) {
    *op_attribute->mutable_mirrored_signature() = *mirrored_signature_;
  } else {
    op_attribute->clear_mirrored_signature();
  }
  if (input_index2logical_blob_desc_) {
    JUST(FillLogicalBlobDescSignature(
        input_bns(), input_index2logical_blob_desc_,
        op_attribute->mutable_logical_blob_desc_signature()->mutable_bn_in_op2blob_desc()));
  }
  if (output_index2logical_blob_desc_) {
    JUST(FillLogicalBlobDescSignature(
        output_bns(), output_index2logical_blob_desc_,
        op_attribute->mutable_logical_blob_desc_signature()->mutable_bn_in_op2blob_desc()));
  }
  if (op_parallel_desc_) {
    *op_attribute->mutable_parallel_conf_signature()->mutable_op_parallel_conf() =
        op_parallel_desc_->parallel_conf();
  }
  if (bn2parallel_desc_) {
    auto* map = op_attribute->mutable_parallel_conf_signature()->mutable_bn_in_op2parallel_conf();
    for (const auto& pair : *bn2parallel_desc_) {
      (*map)[pair.first] = pair.second->parallel_conf();
    }
  }
  if (op_parallel_desc_ && bn2parallel_desc_) {
    if (op_conf().scope_symbol_id() != 0) {
      const auto& scope_storage = *Global<symbol::Storage<Scope>>::Get();
      const auto& scope = JUST(scope_storage.MaybeGet(op_conf().scope_symbol_id()));
      int64_t parallel_desc_symbol_id = JUST(scope.GetParallelDescSymbolId(op_conf()));
      auto* parallel_signature = op_attribute->mutable_parallel_signature();
      parallel_signature->set_op_parallel_desc_symbol_id(parallel_desc_symbol_id);
      auto* symbol_map = parallel_signature->mutable_bn_in_op2parallel_desc_symbol_id();
      for (const auto& pair : *bn2parallel_desc_) {
        if (*pair.second == *op_parallel_desc_) {
          (*symbol_map)[pair.first] = parallel_desc_symbol_id;
        } else {
          (*symbol_map)[pair.first] =
              (*Global<std::shared_ptr<ForeignCallback>>::Get())
                  ->MakeParallelDescSymbol(
                      std::make_shared<cfg::ParallelConf>(pair.second->parallel_conf()));
        }
      }
      for (const auto& tbn : tmp_bns()) { (*symbol_map)[tbn] = parallel_desc_symbol_id; }
    }
  }
  return Maybe<void>::Ok();
}

LogicalBlobId GenLogicalBlobId(const std::string& lbn) {
  LogicalBlobId lbi;
  size_t pos = lbn.find('/');
  CHECK_NE(pos, std::string::npos) << "lbn: " << lbn;
  lbi.set_op_name(lbn.substr(0, pos));
  std::string blob_name_with_hit = lbn.substr(pos + 1);
  size_t vbar_pos = blob_name_with_hit.rfind('|');
  std::string blob_name_with_split_hit = blob_name_with_hit.substr(0, vbar_pos);
  size_t split_pos = blob_name_with_split_hit.rfind(':');
  lbi.set_blob_name(blob_name_with_split_hit.substr(0, split_pos));
  return lbi;
}

Maybe<bool> GetSbpParallelInLbnOrNothing(const std::string& lbn, SbpParallel* sbp) {
  size_t vbar_pos = lbn.rfind('|');
  std::string lbn_with_split_hint = lbn.substr(0, vbar_pos);
  size_t pos = lbn_with_split_hint.rfind(':');
  CHECK_NE(pos, lbn_with_split_hint.length() - 1);
  if (pos == std::string::npos) { return false; }
  std::string split_hint = lbn_with_split_hint.substr(pos + 1);
  if (split_hint[0] == 'S') {
    std::string axis_str = split_hint.substr(1);
    CHECK_OR_RETURN(IsStrInt(axis_str));
    sbp->mutable_split_parallel()->set_axis(oneflow_cast<int64_t>(axis_str));
  } else if (split_hint[0] == 'B') {
    sbp->mutable_broadcast_parallel();
  } else {
    return Error::CheckFailedError()
           << "split hint only support 'S' or 'B', but get:" << split_hint[0];
  }
  return true;
}

Maybe<bool> ParseDisableBoxingFlag(const std::string& lbn_with_hint, bool* disable_boxing) {
  size_t pos = lbn_with_hint.rfind('|');
  if (pos == std::string::npos) { return false; }
  CHECK_NE(pos, lbn_with_hint.length() - 1);
  std::string disable_boxing_str = lbn_with_hint.substr(pos + 1);
  CHECK_OR_RETURN(IsStrInt(disable_boxing_str));
  *disable_boxing = oneflow_cast<int64_t>(disable_boxing_str);
  return true;
}

std::string GetInputLbnInOpCustomizedConf(const OperatorConf& op_conf,
                                          const std::string& fd_name_may_have_idx) {
  const PbMessage& msg = GetMessageInPbMessage(op_conf, op_conf.op_type_case());
  const PbMessage* msg_ptr = &msg;
  const UserOpConf* user_conf = dynamic_cast<const UserOpConf*>(msg_ptr);
  if (user_conf) {
    std::pair<std::string, int32_t> pair = GetFieldNameAndIndex4StrVal(fd_name_may_have_idx);
    if (user_conf->input().find(pair.first) != user_conf->input().end()) {
      return user_conf->input().at(pair.first).s(pair.second);
    } else {
      LOG(WARNING) << "cannot find input arg val in user op conf. (arg_name = " << pair.first
                   << ", id = " << std::to_string(pair.second) << ")";
      return "";
    }
  } else {
    return GetStrValInPbFdOrPbRpf(msg, fd_name_may_have_idx);
  }
}

// return old value
std::string ReplaceInputLbnInOpTypeConf(PbMessage* msg, const std::string& fd_name_may_have_idx,
                                        const std::string& new_val) {
  UserOpConf* user_conf = dynamic_cast<UserOpConf*>(msg);
  std::string old_val;
  if (user_conf) {
    std::pair<std::string, int32_t> pair = GetFieldNameAndIndex4StrVal(fd_name_may_have_idx);
    CHECK(user_conf->input().find(pair.first) != user_conf->input().end())
        << "cannot find input arg val in user op conf. (arg_name = " << pair.first
        << ", id = " << std::to_string(pair.second) << ")\n"
        << " new lbn = " << new_val;
    old_val = user_conf->input().at(pair.first).s(pair.second);
    (*(user_conf->mutable_input()))[pair.first].set_s(pair.second, new_val);
  } else {
    old_val = ReplaceStrValInPbFdOrPbRpf(msg, fd_name_may_have_idx, new_val);
  }
  return old_val;
}

std::string ReplaceInputLbnInOpCustomizedConf(OperatorConf* op_conf,
                                              const std::string& fd_name_may_have_idx,
                                              const std::string& new_val) {
  PbMessage* op_type_conf = MutableMessageInPbMessage(op_conf, op_conf->op_type_case());
  return ReplaceInputLbnInOpTypeConf(op_type_conf, fd_name_may_have_idx, new_val);
}

bool operator==(const OperatorConf& lhs, const OperatorConf& rhs) {
  return PbMd().Equals(lhs, rhs);
}

namespace {

Maybe<void> InferOpOutSbpParallel(
    Operator* op, const OpNodeSignature& upstream_signature,
    const std::function<const BlobDesc&(const std::string&)>& ConstBlobDesc4Ibn,
    const SbpSignature& sbp_sig_conf, const ParallelDesc& parallel_desc) {
  const auto& SbpParallel4Ibn = [&](const std::string& ibn) -> const SbpParallel* {
    const auto& map = upstream_signature.sbp_signature().bn_in_op2sbp_parallel();
    return &map.at(ibn);
  };
  HashMap<std::string, SbpInferHint> ibn2sbp_infer_hint;
  for (const std::string& ibn : op->input_bns()) {
    const ParallelDesc* pd = &parallel_desc;
    const BlobDesc* logical_blob_desc = &ConstBlobDesc4Ibn(ibn);
    const SbpParallel* sbp_parallel = SbpParallel4Ibn(ibn);
    ibn2sbp_infer_hint.emplace(ibn, SbpInferHint(pd, logical_blob_desc, sbp_parallel));
  }
  SbpSignature sbp_signature;
  JUST(op->InferSbpSignature(&sbp_signature, sbp_sig_conf, ibn2sbp_infer_hint));
  JUST(op->FillSbpSignature(sbp_signature));
  return Maybe<void>::Ok();
}

Maybe<void> InferMirroredSignature(Operator* op, const OpNodeSignature& upstream_signature,
                                   bool is_mirrored, const ParallelDesc& parallel_desc) {
  HashMap<std::string, MirroredSigInferHint> ibn2mirrored_sig_infer_hint;
  for (const std::string& ibn : op->input_bns()) {
    const auto& map = upstream_signature.mirrored_signature().bn_in_op2opt_mirrored_parallel();
    const auto& opt_mirrored_parallel = map.at(ibn);
    ibn2mirrored_sig_infer_hint.emplace(
        ibn, MirroredSigInferHint(&parallel_desc, opt_mirrored_parallel.has_mirrored_parallel()));
  }
  const auto& MirroredSigInferHint4Ibn =
      [&](const std::string& ibn) -> Maybe<const MirroredSigInferHint*> {
    const auto& iter = ibn2mirrored_sig_infer_hint.find(ibn);
    CHECK_OR_RETURN(iter != ibn2mirrored_sig_infer_hint.end())
        << "input blob not found. ibn: " << ibn;
    return &iter->second;
  };
  JUST(op->InferMirroredSignatureIf(MirroredSigInferHint4Ibn, is_mirrored, parallel_desc));
  return Maybe<void>::Ok();
}

Maybe<void> CheckOpInputSignature(const Operator& op, const OpNodeSignature& upstream_signature) {
  for (const auto& ibn : op.input_bns()) {
    {
      CHECK_OR_RETURN(upstream_signature.has_logical_blob_desc_signature());
      const auto& map = upstream_signature.logical_blob_desc_signature().bn_in_op2blob_desc();
      CHECK_OR_RETURN(map.find(ibn) != map.end());
    }
    {
      CHECK_OR_RETURN(upstream_signature.has_sbp_signature());
      const auto& map = upstream_signature.sbp_signature().bn_in_op2sbp_parallel();
      CHECK_OR_RETURN(map.find(ibn) != map.end());
    }
    {
      CHECK_OR_RETURN(upstream_signature.has_mirrored_signature());
      const auto& map = upstream_signature.mirrored_signature().bn_in_op2opt_mirrored_parallel();
      CHECK_OR_RETURN(map.find(ibn) != map.end());
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<Operator> ConstructAndInferOp(const OperatorConf& op_conf,
                                    const OpNodeSignature& upstream_signature, const Scope& scope) {
  const auto& parallel_desc = JUST(scope.GetParallelDesc(op_conf));
  bool is_mirrored = scope.opt_mirrored_parallel_conf().has_mirrored_parallel();
  const auto& op = ConstructOp(op_conf);
  JUST(CheckOpInputSignature(*op, upstream_signature));
  JUST(op->FillOpParallelDesc(parallel_desc));
  HashMap<std::string, std::unique_ptr<BlobDesc>> bn_in_op2blob_desc;
  for (const auto& ibn : op->input_bns()) {
    const auto& map = upstream_signature.logical_blob_desc_signature().bn_in_op2blob_desc();
    bn_in_op2blob_desc[ibn].reset(new BlobDesc(map.at(ibn)));
  }
  const auto& ConstBlobDesc4Ibn = [&](const std::string& ibn) -> const BlobDesc& {
    return *bn_in_op2blob_desc.at(ibn);
  };
  JUST(op->FillLogicalInBlobDesc(ConstBlobDesc4Ibn));
  // infer is_mirrored
  JUST(InferMirroredSignature(op.get(), upstream_signature, is_mirrored, parallel_desc));
  SbpSignature sbp_sig_conf;
  // iner sbp
  JUST(InferOpOutSbpParallel(op.get(), upstream_signature, ConstBlobDesc4Ibn, sbp_sig_conf,
                             parallel_desc));
  // infer logical blob_desc
  JUST(op->InferLogicalOutBlobDescsIf());
  return op;
}

namespace {

Maybe<Shape> Get1dHierarchyPhysicalShape(const Shape& logical_shape,
                                         const SbpParallel& sbp_parallel,
                                         const int64_t parallel_num, const int64_t parallel_id) {
  std::shared_ptr<Shape> physical = std::make_shared<Shape>(logical_shape);
  if (sbp_parallel.has_split_parallel()) {
    const int64_t axis = sbp_parallel.split_parallel().axis();
    CHECK_GE_OR_RETURN(logical_shape.At(axis), parallel_num);
    const BalancedSplitter bs(logical_shape.At(axis), parallel_num);
    physical->Set(axis, bs.At(parallel_id).size());
  } else if (sbp_parallel.has_broadcast_parallel() || sbp_parallel.has_partial_sum_parallel()) {
    // do nothing
  } else {
    UNIMPLEMENTED();
  }
  return physical;
}

Maybe<Shape> GetNdHierarchyPhysicalShape(const Shape& logical_shape,
                                         const ParallelDistribution& parallel_distribution,
                                         const Shape& parallel_hierarchy) {
  std::shared_ptr<Shape> physical = std::make_shared<Shape>(logical_shape);
  FOR_RANGE(int64_t, i, 0, parallel_hierarchy.NumAxes()) {
    const SbpParallel& sbp_parallel = parallel_distribution.sbp_parallel(i);
    if (sbp_parallel.has_split_parallel()) {
      const int64_t split_axis = sbp_parallel.split_parallel().axis();
      CHECK_EQ_OR_RETURN(physical->At(split_axis) % parallel_hierarchy.At(i), 0);
      physical->Set(split_axis, physical->At(split_axis) / parallel_hierarchy.At(i));
    }
  }
  return physical;
}

}  // namespace

Maybe<Shape> GetPhysicalShape(const Shape& logical_shape,
                              const ParallelDistribution& parallel_distribution,
                              const ParallelDesc& parallel_desc,
                              const ParallelContext& parallel_ctx) {
  CHECK_LT_OR_RETURN(parallel_ctx.parallel_id(), parallel_desc.hierarchy()->elem_cnt());
  CHECK_EQ_OR_RETURN(parallel_desc.hierarchy()->NumAxes(),
                     parallel_distribution.sbp_parallel_size());
  if (parallel_desc.hierarchy()->NumAxes() == 1) {
    return Get1dHierarchyPhysicalShape(logical_shape, parallel_distribution.sbp_parallel(0),
                                       parallel_desc.hierarchy()->elem_cnt(),
                                       parallel_ctx.parallel_id());
  } else {
    return GetNdHierarchyPhysicalShape(logical_shape, parallel_distribution,
                                       *parallel_desc.hierarchy());
  }
}

}  // namespace oneflow
