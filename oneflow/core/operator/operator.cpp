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
#include "oneflow/core/eager/eager_symbol_storage.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/job/mirrored_sig_infer_hint.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/operator/operator.h"

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

std::shared_ptr<Operator> CheckAndConstructOp(const OperatorConf& op_conf,
                                              const JobDesc* job_desc) {
  Operator* rptr = NewObj<Operator>(op_conf.op_type_case(), op_conf);
  DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(op_conf.device_tag()));
  if (IsCpuOnly(op_conf)) { CHECK_EQ(device_type, DeviceType::kCPU); }
  rptr->Init(op_conf, job_desc);
  return std::shared_ptr<Operator>(rptr);
}

}  // namespace

void Operator::Init(const OperatorConf& op_conf, const JobDesc* conf_job_desc) {
  job_desc_ = conf_job_desc;
  OperatorConf* this_op_conf = op_attribute_.mutable_op_conf();
  *this_op_conf = op_conf;
  if (has_job_desc() && job_desc().IsPredict()) { this_op_conf->set_trainable(false); }
  if (has_job_desc() && this_op_conf->has_enable_cudnn() == false) {
    this_op_conf->set_enable_cudnn(job_desc().EnableCudnn());
  }
  InitFromOpConf();
}

LogicalNode* Operator::NewProperLogicalNode() const { return new NormalForwardLogicalNode; }

const LogicalBlobId& Operator::BnInOp2Lbi(const std::string& bn_in_op) const {
  return op_attribute_.arg_signature().bn_in_op2lbi().at(bn_in_op);
}

LogicalBlobId* Operator::MutBnInOp2Lbi(const std::string& bn_in_op) {
  auto it = op_attribute_.mutable_arg_signature()->mutable_bn_in_op2lbi()->find(bn_in_op);
  if (it == op_attribute_.mutable_arg_signature()->mutable_bn_in_op2lbi()->end()) {
    return nullptr;
  } else {
    return &(it->second);
  }
}

DeviceType Operator::device_type() const {
  DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(op_attribute_.op_conf().device_tag()));
  return device_type;
}

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
  const auto& iter = lbi2obn_.find(lbi);
  CHECK_OR_RETURN(iter != lbi2obn_.end())
      << "no logical blob id found. lbn: " << lbi.op_name() << "/" << lbi.blob_name();
  return &iter->second;
}

Maybe<void> Operator::InferParallelSignatureIf() {
  if (op_conf().scope_symbol_id() == 0) { return Maybe<void>::Ok(); }
  return InferParallelSignature();
}

Maybe<void> Operator::InferParallelSignature() {
  const auto& scope_storage = *Global<vm::SymbolStorage<Scope>>::Get();
  const auto& scope = JUST(scope_storage.MaybeGet(op_conf().scope_symbol_id()));
  int64_t parallel_desc_symbol_id = JUST(scope.GetParallelDescSymbolId(op_conf()));
  auto* parallel_signature = op_attribute_.mutable_parallel_signature();
  parallel_signature->set_op_parallel_desc_symbol_id(parallel_desc_symbol_id);
  auto* map = parallel_signature->mutable_bn_in_op2parallel_desc_symbol_id();
  for (const auto& ibn : input_bns()) { (*map)[ibn] = parallel_desc_symbol_id; }
  for (const auto& obn : output_bns()) { (*map)[obn] = parallel_desc_symbol_id; }
  for (const auto& tbn : tmp_bns()) { (*map)[tbn] = parallel_desc_symbol_id; }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const std::function<Maybe<const OptInt64*>(const std::string&)>& BatchAxis4Ibn,
    const ParallelDesc& parallel_desc) const {
  ParallelContext parallel_ctx;
  parallel_ctx.set_parallel_id(0);
  parallel_ctx.set_parallel_num(1);
  SbpSignature sbp_signature;
  auto* map = sbp_signature.mutable_bn_in_op2sbp_parallel();
  for (const auto& ibn : input_bns()) { (*map)[ibn].mutable_split_parallel()->set_axis(0); }
  for (const auto& obn : output_bns()) { (*map)[obn].mutable_split_parallel()->set_axis(0); }
  return InferOutBlobDescsIf(BlobDesc4BnInOp, &parallel_ctx, &sbp_signature, [](OpContext*) {});
}

Maybe<void> Operator::InferBlobDescsIf(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  return InferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, sbp_signature, EnrollOpCtx);
}

Maybe<void> Operator::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  return InferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, sbp_signature);
}

Maybe<void> Operator::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  return InferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

Maybe<void> Operator::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  UNIMPLEMENTED() << typeid(*this).name();
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferOutBlobDescsIf(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  return InferOutBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, sbp_signature, EnrollOpCtx);
}

Maybe<void> Operator::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  // TODO() separate InferOut and InferTmp
  // At present, only conv_op infer out blob separately
  return InferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, sbp_signature, EnrollOpCtx);
}

Maybe<void> Operator::FillLogicalBlobDescSignature(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& BlobDesc4BnInOp) {
  auto* map = op_attribute_.mutable_logical_blob_desc_signature()->mutable_bn_in_op2blob_desc();
  for (const auto& ibn : input_bns()) { JUST(BlobDesc4BnInOp(ibn)).ToProto(&(*map)[ibn]); }
  for (const auto& obn : output_bns()) { JUST(BlobDesc4BnInOp(obn)).ToProto(&(*map)[obn]); }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferOutParallelDescIf(
    std::function<ParallelDesc*(const std::string&)> ParallelDesc4Obn,
    std::function<const BlobDesc*(const std::string&)> LogicalBlobDesc4Ibn,
    const ParallelDesc& op_parallel_desc, const SbpSignature* sbp_signature) const {
  return InferOutParallelDesc(ParallelDesc4Obn, LogicalBlobDesc4Ibn, op_parallel_desc,
                              sbp_signature);
}

Maybe<void> Operator::InferOutParallelDesc(
    std::function<ParallelDesc*(const std::string&)> ParallelDesc4Obn,
    std::function<const BlobDesc*(const std::string&)> LogicalBlobDesc4Ibn,
    const ParallelDesc& op_parallel_desc, const SbpSignature* sbp_signature) const {
  for (const auto& obn : output_bns()) { *ParallelDesc4Obn(obn) = op_parallel_desc; }
  return Maybe<void>::Ok();
}

Maybe<void> Operator::InferOutputBlobTimeShapeIf(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  if (!input_bns().empty()) {
    const int64_t first_input_time_shape_elem_cnt =
        GetTimeShape4BnInOp(input_bns().Get(0))->elem_cnt();
    FOR_RANGE(int64_t, i, 1, input_bns().size()) {
      CHECK_EQ_OR_RETURN(GetTimeShape4BnInOp(input_bns().Get(i))->elem_cnt(),
                         first_input_time_shape_elem_cnt);
    }
  }
  return InferOutputBlobTimeShape(GetTimeShape4BnInOp, parallel_ctx, time_shape);
}

Maybe<void> Operator::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp, const ParallelContext*,
    Shape* time_shape) const {
  if (input_bns().empty() == false) {
    *time_shape = *GetTimeShape4BnInOp(input_bns().Get(0));
  } else {
    CHECK_OR_RETURN(has_job_desc());
    *time_shape = Shape({job_desc().TotalBatchNum(), job_desc().NumOfPiecesInBatch()});
  }
  return Maybe<void>::Ok();
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
  for (const std::string& bn_in_op : const_buf_bns()) { Handler(bn_in_op); }
  for (const std::string& bn_in_op : tmp_bns()) { Handler(bn_in_op); }
}

Maybe<void> Operator::InferSbpSignatureIf(
    const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) {
  if (parallel_desc.parallel_num() == 1) {
    auto* bn2sbp = mut_sbp_signature()->mutable_bn_in_op2sbp_parallel();
    for (const auto& ibn : input_bns()) { (*bn2sbp)[ibn].mutable_split_parallel()->set_axis(0); }
    for (const auto& obn : output_bns()) { (*bn2sbp)[obn].mutable_split_parallel()->set_axis(0); }
  } else if (parallel_desc.parallel_num() > 1) {
    return InferSbpSignature(mut_sbp_signature(), sbp_sig_conf, CalcOrderValue4SbpSig,
                             SbpInferHint4Ibn, parallel_desc);
  } else {
    UNIMPLEMENTED();
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
  SbpSignatureList sbp_sig_list;
  JUST(GetSbpSignaturesIf(LogicalBlobDesc4Ibn, parallel_desc, &sbp_sig_list));
  // filter sbp signatures by sbp signature conf
  SbpSignatureList filtered_sbp_sigs_by_conf;
  FilterSbpSignatureList(sbp_sig_list, sbp_sig_conf, &filtered_sbp_sigs_by_conf);
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
  CHECK_OR_RETURN(op_attribute_.has_sbp_signature()) << "sbp signature not infered";
  return &op_attribute_.sbp_signature();
}

Maybe<const SbpParallel*> Operator::SbpParallel4BnInOp(const std::string& bn_in_op) const {
  CHECK_OR_RETURN(op_attribute_.has_sbp_signature()) << "sbp signature not infered";
  const auto& map = op_attribute_.sbp_signature().bn_in_op2sbp_parallel();
  const auto& iter = map.find(bn_in_op);
  CHECK_OR_RETURN(iter != map.end()) << "blob_name " << bn_in_op << " not found in sbp signature";
  return &iter->second;
}

Maybe<const OptInt64*> Operator::BatchAxis4BnInOp(const std::string& bn_in_op) const {
  CHECK_OR_RETURN(op_attribute_.has_batch_axis_signature()) << "batch axis signature not infered";
  const auto& map = op_attribute_.batch_axis_signature().bn_in_op2batch_axis();
  const auto& iter = map.find(bn_in_op);
  CHECK_OR_RETURN(iter != map.end())
      << "blob_name " << bn_in_op << " not found in batch axis signature";
  return &iter->second;
}

Maybe<const OptMirroredParallel*> Operator::OptMirroredParallel4BnInOp(
    const std::string& bn_in_op) const {
  CHECK_OR_RETURN(op_attribute_.has_mirrored_signature()) << "mirrored signature not infered";
  const auto& map = op_attribute_.mirrored_signature().bn_in_op2opt_mirrored_parallel();
  const auto& iter = map.find(bn_in_op);
  CHECK_OR_RETURN(iter != map.end())
      << "blob_name " << bn_in_op << " not found in mirrored signature";
  return &iter->second;
}

OptMirroredParallel* Operator::MutOptMirroredParallel(const std::string& bn_in_op) {
  auto* map = op_attribute_.mutable_mirrored_signature()->mutable_bn_in_op2opt_mirrored_parallel();
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
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx,
    std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp,
    const ParallelDesc* parallel_desc) const {
  auto* dtype_signature = kernel_conf->mutable_dtype_signature();
  for (const std::string& ibn : input_bns()) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(ibn);
    if (blob_desc == nullptr) { continue; }
    (*dtype_signature->mutable_name2dtype())[ibn] = blob_desc->data_type();
  };
  *(kernel_conf->mutable_op_attribute()) = op_attribute_;
  if (HasBlobDescWithField(GetBlobDesc4BnInOp, output_bns(), [](const BlobDesc* blob_desc) {
        return blob_desc->header_is_opaque();
      })) {
    kernel_conf->set_need_do_opaque_header(true);
  } else {
    if (HasBlobDescWithField(GetBlobDesc4BnInOp, output_bns(),
                             [](const BlobDesc* blob_desc) { return blob_desc->is_dynamic(); })) {
      kernel_conf->set_need_do_shape(true);
    }
    if (HasBlobDescWithField(GetBlobDesc4BnInOp, output_bns(), [](const BlobDesc* blob_desc) {
          return blob_desc->is_tensor_list();
        })) {
      kernel_conf->set_need_do_tensor_list(true);
    }
  }

  {
    DataType data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, output_bns());
    if (data_type == DataType::kInvalidDataType) {
      data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, input_bns());
    }
    kernel_conf->set_data_type(data_type);
  }

  VirtualGenKernelConf(GetBlobDesc4BnInOp, parallel_ctx, kernel_conf, op_ctx,
                       LogicalBlobDesc4BnInOp, parallel_desc);
}

void Operator::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx,
    std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp,
    const ParallelDesc* parallel_desc) const {
  VirtualGenKernelConf(GetBlobDesc4BnInOp, parallel_ctx, kernel_conf, op_ctx,
                       LogicalBlobDesc4BnInOp);
}

void Operator::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx,
    std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const {
  VirtualGenKernelConf(GetBlobDesc4BnInOp, parallel_ctx, kernel_conf, op_ctx);
}

void Operator::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  VirtualGenKernelConf(GetBlobDesc4BnInOp, parallel_ctx, kernel_conf);
}

int64_t Operator::cudnn_buf_limit_byte() const {
  int64_t cudnn_buf_limit_mbyte = 0;
  if (op_conf().has_cudnn_buf_limit_mbyte()) {
    cudnn_buf_limit_mbyte = op_conf().cudnn_buf_limit_mbyte();
  } else {
    CHECK(has_job_desc());
    cudnn_buf_limit_mbyte = job_desc().cudnn_buf_limit_mbyte();
  }
  return cudnn_buf_limit_mbyte * 1024 * 1024;
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
LogicalBlobId Operator::cbbn2lbi(const std::string& const_buf_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(const_buf_bn);
  return ret;
}

void Operator::EnrollTmpBn(const std::string& tbn) {
  *(mut_tmp_bns()->Add()) = tbn;
  CHECK(mut_bn_in_op2lbi()->insert({tbn, tbn2lbi(tbn)}).second);
}

InputBlobModifier* Operator::EnrollInputBn(const std::string& ibn, bool has_diff) {
  LogicalBlobId lbi = lbi4ibn(ibn);
  auto* map = op_attribute_.mutable_arg_modifier_signature()->mutable_ibn2input_blob_modifier();
  CHECK(map->insert({ibn, InputBlobModifier()}).second);
  *(mut_input_bns()->Add()) = ibn;
  CHECK(mut_bn_in_op2lbi()->insert({ibn, lbi}).second);
  auto* ret = MutInputBlobModifier4Ibn(ibn);
  ret->set_requires_grad(has_diff);
  return ret;
}

const InputBlobModifier& Operator::InputBlobModifier4Ibn(const std::string& ibn) const {
  return op_attribute_.arg_modifier_signature().ibn2input_blob_modifier().at(ibn);
}

const OutputBlobModifier& Operator::OutputBlobModifier4Obn(const std::string& obn) const {
  return op_attribute_.arg_modifier_signature().obn2output_blob_modifier().at(obn);
}

InputBlobModifier* Operator::MutInputBlobModifier4Ibn(const std::string& ibn) {
  auto* map = op_attribute_.mutable_arg_modifier_signature()->mutable_ibn2input_blob_modifier();
  return &map->at(ibn);
}

OutputBlobModifier* Operator::MutOutputBlobModifier4Obn(const std::string& obn) {
  auto* map = op_attribute_.mutable_arg_modifier_signature()->mutable_obn2output_blob_modifier();
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

void Operator::EmplaceLbi2Obn(const LogicalBlobId& lbi, const std::string& obn) {
  CHECK(lbi2obn_.emplace(lbi, obn).second);
}

OutputBlobModifier* Operator::EnrollOutputBn(const std::string& obn, bool has_diff) {
  LogicalBlobId lbi = lbi4obn(obn);
  EmplaceLbi2Obn(lbi, obn);
  auto* map = op_attribute_.mutable_arg_modifier_signature()->mutable_obn2output_blob_modifier();
  CHECK(map->insert({obn, OutputBlobModifier()}).second);
  *(mut_output_bns()->Add()) = obn;
  CHECK(mut_bn_in_op2lbi()->insert({obn, lbi}).second);
  auto* ret = MutOutputBlobModifier4Obn(obn);
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

void Operator::EnrollConstBufBn(const std::string& cbbn) {
  *(mut_const_buf_bns()->Add()) = cbbn;
  CHECK(mut_bn_in_op2lbi()->insert({cbbn, cbbn2lbi(cbbn)}).second);
}

void Operator::StrFieldTolower(const std::string& field_name) {
  std::string field_val = GetValFromCustomizedConf<std::string>(field_name);
  std::transform(field_val.begin(), field_val.end(), field_val.begin(), ::tolower);
  SetValInCustomizedConf(field_name, field_val);
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
  auto* ptr = NewObj<CpuOnly>(op_type_case);
  CHECK(ptr != nullptr) << "op_conf\n" << op_conf.DebugString();
  if (*std::unique_ptr<CpuOnly>(ptr)) { return true; }
  if (!op_conf.has_user_conf()) { return false; }
  auto* registration_val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_conf.user_conf().op_type_name());
  CHECK_NOTNULL(registration_val);
  return registration_val->cpu_only_supported;
}

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf, DeviceType device_type,
                                      const JobDesc* job_desc) {
  OperatorConf dev_op_conf = op_conf;
  dev_op_conf.set_device_tag(CHECK_JUST(DeviceTag4DeviceType(device_type)));
  return CheckAndConstructOp(dev_op_conf, job_desc);
}

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf, const JobDesc* job_desc) {
  if (IsCpuOnly(op_conf)) { return ConstructOp(op_conf, DeviceType::kCPU, job_desc); }
  return CheckAndConstructOp(op_conf, job_desc);
}

void EraseEmptyBnInVec(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                       PbRpf<std::string>* bns) {
  size_t idx_available = 0;
  for (size_t i = 0; i < bns->size(); ++i) {
    if (GetBlobDesc4BnInOp((*bns)[i])) {
      if (i != idx_available) { (*bns)[idx_available] = (*bns)[i]; }
      ++idx_available;
    }
  }
  bns->erase(bns->begin() + idx_available, bns->end());
}

Maybe<void> Operator::InferBatchAxisIf(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    std::function<Maybe<const OptInt64*>(const std::string&)> BatchAxis4Ibn) {
  auto* map = op_attribute_.mutable_batch_axis_signature()->mutable_bn_in_op2batch_axis();
  for (const auto& ibn : input_bns()) { (*map)[ibn] = *JUST(BatchAxis4Ibn(ibn)); }
  const auto& BatchAxis4BnInOp = [&](const std::string& bn_in_op) { return &(*map)[bn_in_op]; };
  return InferBatchAxis(LogicalBlobDesc4Ibn, BatchAxis4BnInOp);
}

Maybe<void> Operator::NaiveInferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  if (output_bns().empty()) { return Maybe<void>::Ok(); }
  CHECK_GT_OR_RETURN(input_bns().size(), 0);
  CHECK_EQ_OR_RETURN(output_bns().size(), 1);
  const OptInt64* batch_axis = nullptr;
  for (const auto& ibn : input_bns()) {
    const OptInt64* const cur_ibn_batch_axis = BatchAxis4BnInOp(ibn);
    if (cur_ibn_batch_axis->has_value() == false) { continue; }
    if (batch_axis) {
      CHECK_OR_RETURN(*batch_axis == *cur_ibn_batch_axis);
    } else {
      batch_axis = cur_ibn_batch_axis;
    }
  }
  OptInt64 no_batch_axis;
  if (batch_axis == nullptr) { batch_axis = &no_batch_axis; }
  *BatchAxis4BnInOp(SoleObn()) = *batch_axis;
  return Maybe<void>::Ok();
}

Symbol<OperatorConf> Operator::GetOpConfWithoutOpNameAndLbn() const {
  OperatorConf op_conf(this->op_conf());
  op_conf.set_name("undefined-op-name");
  PbMessage* op_type_conf = MutableMessageInPbMessage(&op_conf, op_conf.op_type_case());
  for (const auto& ibn : input_bns()) {
    if (!HasStrFieldInPbFdOrPbRpf(*op_type_conf, ibn)) { continue; }
    const std::string& lbn = GetInputLbnInOpCustomizedConf(*op_type_conf, ibn);
    ReplaceInputLbnInOpCustomizedConf(op_type_conf, ibn, lbn, "undefined-op-name/undefined-ibn");
  }
  return SymbolOf(op_conf);
}

std::shared_ptr<OpAttribute> Operator::GetOpAttributeWithoutOpNameAndLbn() const {
  auto op_attribute = std::make_shared<OpAttribute>(op_attribute_);
  op_attribute->mutable_sbp_signature();
  *op_attribute->mutable_op_conf() = *GetOpConfWithoutOpNameAndLbn();
  return op_attribute;
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

Maybe<void> InferOpSbpSignature(
    Operator* op, const SbpSignature& sbp_sig_conf, const ParallelDesc& parallel_desc,
    const HashMap<std::string, SbpInferHint>& ibn2sbp_infer_hint,
    std::function<Maybe<const OptInt64*>(const std::string&)> BatchAxis4BnInOp) {
  auto SbpInferHint4Ibn = [&](const std::string& ibn) -> Maybe<const SbpInferHint*> {
    auto it = ibn2sbp_infer_hint.find(ibn);
    if (it == ibn2sbp_infer_hint.end()) {
      return Error::CheckFailedError()
             << "cannot find corresponding SbpInferHint for input_blob_name : " << ibn;
    }
    return &(it->second);
  };
  std::function<int32_t(const SbpSignature&)> CalcOrderValue4SbpSig;
  auto OrderValue4HasBatchAxis = [&](const std::string& bn,
                                     const SbpParallel& sbp_parallel) -> int32_t {
    const auto& batch_axis = *CHECK_JUST(BatchAxis4BnInOp(bn));
    return -1
           * (batch_axis.has_value() && sbp_parallel.has_split_parallel()
              && sbp_parallel.split_parallel().axis() == batch_axis.value());
  };
  auto OrderValue4HasNoBatchAxis = [&](const std::string& ibn,
                                       const SbpParallel& sbp_parallel) -> int32_t {
    const auto& batch_axis = *CHECK_JUST(BatchAxis4BnInOp(ibn));
    return -2
           * (batch_axis.has_value() == false
              && CHECK_JUST(SbpInferHint4Ibn(ibn))->sbp_parallel().has_split_parallel() == false
              && sbp_parallel.has_split_parallel() == false);
  };
  auto OrderValue4SbpHint = [&](const std::string& ibn,
                                const SbpParallel& sbp_parallel) -> int32_t {
    return -3 * (CHECK_JUST(SbpInferHint4Ibn(ibn))->sbp_parallel() == sbp_parallel);
  };
  if (sbp_sig_conf.bn_in_op2sbp_parallel().empty()) {
    CalcOrderValue4SbpSig = [&](const SbpSignature& sbp_signature) -> int32_t {
      int32_t order_value = 0;
      for (const auto& ibn : op->input_bns()) {
        const auto& sbp_parallel_it = sbp_signature.bn_in_op2sbp_parallel().find(ibn);
        CHECK(sbp_parallel_it != sbp_signature.bn_in_op2sbp_parallel().end());
        order_value += OrderValue4HasBatchAxis(ibn, sbp_parallel_it->second);
        order_value += OrderValue4HasNoBatchAxis(ibn, sbp_parallel_it->second);
        order_value += OrderValue4SbpHint(ibn, sbp_parallel_it->second);
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
  JUST(op->InferSbpSignatureIf(sbp_sig_conf, CalcOrderValue4SbpSig, SbpInferHint4Ibn,
                               parallel_desc));
  return Maybe<void>::Ok();
}

std::string GetInputLbnInOpCustomizedConf(const PbMessage& msg,
                                          const std::string& fd_name_may_have_idx) {
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

void ReplaceInputLbnInOpCustomizedConf(PbMessage* msg, const std::string& fd_name_may_have_idx,
                                       const std::string& old_val, const std::string& new_val) {
  UserOpConf* user_conf = dynamic_cast<UserOpConf*>(msg);
  if (user_conf) {
    std::pair<std::string, int32_t> pair = GetFieldNameAndIndex4StrVal(fd_name_may_have_idx);
    CHECK(user_conf->input().find(pair.first) != user_conf->input().end())
        << "cannot find input arg val in user op conf. (arg_name = " << pair.first
        << ", id = " << std::to_string(pair.second) << ")\n"
        << "old lbn = " << old_val << " new lbn = " << new_val;
    CHECK_EQ(user_conf->input().at(pair.first).s(pair.second), old_val);
    (*(user_conf->mutable_input()))[pair.first].set_s(pair.second, new_val);
  } else {
    ReplaceStrValInPbFdOrPbRpf(msg, fd_name_may_have_idx, old_val, new_val);
  }
}

bool operator==(const OperatorConf& lhs, const OperatorConf& rhs) {
  return PbMd().Equals(lhs, rhs);
}

namespace {

Maybe<void> InferOpOutSbpParallel(
    Operator* op, const OpNodeSignature& upstream_signature,
    const std::function<const BlobDesc&(const std::string&)>& ConstBlobDesc4Ibn,
    const SbpSignature& sbp_sig_conf, const ParallelDesc& parallel_desc) {
  const auto& BatchAxis4BnInOp = [&](const std::string& bn_in_op) -> Maybe<const OptInt64*> {
    return op->BatchAxis4BnInOp(bn_in_op);
  };
  const auto& SbpParallel4Ibn = [&](const std::string& ibn) -> const SbpParallel* {
    const auto& map = upstream_signature.sbp_signature().bn_in_op2sbp_parallel();
    return &map.at(ibn);
  };
  HashMap<std::string, SbpInferHint> ibn2sbp_infer_hint;
  for (const std::string& ibn : op->input_bns()) {
    const ParallelDesc* pd = &parallel_desc;
    const BlobDesc* logical_blob_desc = &ConstBlobDesc4Ibn(ibn);
    const SbpParallel* sbp_parallel = SbpParallel4Ibn(ibn);
    const OptInt64* batch_axis = JUST(BatchAxis4BnInOp(ibn));
    ibn2sbp_infer_hint.emplace(ibn, SbpInferHint(pd, logical_blob_desc, sbp_parallel, batch_axis));
  }

  JUST(InferOpSbpSignature(op, sbp_sig_conf, parallel_desc, ibn2sbp_infer_hint, BatchAxis4BnInOp));
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
    {
      CHECK_OR_RETURN(upstream_signature.has_batch_axis_signature());
      const auto& map = upstream_signature.batch_axis_signature().bn_in_op2batch_axis();
      CHECK_OR_RETURN(map.find(ibn) != map.end());
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<Operator> ConstructAndInferOp(const OperatorConf& op_conf,
                                    const OpNodeSignature& upstream_signature, const Scope& scope) {
  const auto& parallel_desc = *JUST(scope.GetParallelDesc(op_conf));
  bool is_mirrored = scope.opt_mirrored_parallel_conf().has_mirrored_parallel();
  const auto& op = ConstructOp(op_conf, JUST(scope.job_desc()));
  JUST(CheckOpInputSignature(*op, upstream_signature));
  HashMap<std::string, std::unique_ptr<BlobDesc>> bn_in_op2blob_desc;
  for (const auto& ibn : op->input_bns()) {
    const auto& map = upstream_signature.logical_blob_desc_signature().bn_in_op2blob_desc();
    bn_in_op2blob_desc[ibn].reset(new BlobDesc(map.at(ibn)));
  }
  const auto& ConstBlobDesc4Ibn = [&](const std::string& ibn) -> const BlobDesc& {
    return *bn_in_op2blob_desc.at(ibn);
  };
  const auto& BatchAxis4Ibn = [&](const std::string& ibn) -> Maybe<const OptInt64*> {
    const auto& map = upstream_signature.batch_axis_signature().bn_in_op2batch_axis();
    const auto& iter = map.find(ibn);
    CHECK_OR_RETURN(iter != map.end());
    return &iter->second;
  };
  // infer batch_axis
  JUST(op->InferBatchAxisIf(ConstBlobDesc4Ibn, BatchAxis4Ibn));
  // infer is_mirrored
  JUST(InferMirroredSignature(op.get(), upstream_signature, is_mirrored, parallel_desc));
  SbpSignature sbp_sig_conf;
  // iner sbp
  JUST(InferOpOutSbpParallel(op.get(), upstream_signature, ConstBlobDesc4Ibn, sbp_sig_conf,
                             parallel_desc));
  const auto& BlobDesc4BnInOp = [&](const std::string& bn_in_op) -> BlobDesc* {
    if (!bn_in_op2blob_desc[bn_in_op]) {
      bn_in_op2blob_desc[bn_in_op].reset(new BlobDesc(DataType::kInvalidDataType));
    }
    return bn_in_op2blob_desc[bn_in_op].get();
  };
  // infer logical blob_desc
  JUST(op->InferLogicalOutBlobDescsIf(BlobDesc4BnInOp, BatchAxis4Ibn, parallel_desc));
  JUST(op->FillLogicalBlobDescSignature([&](const std::string& bn_in_op) -> Maybe<const BlobDesc&> {
    CHECK_OR_RETURN(bn_in_op2blob_desc.find(bn_in_op) != bn_in_op2blob_desc.end());
    return *bn_in_op2blob_desc[bn_in_op];
  }));
  return op;
}

}  // namespace oneflow
