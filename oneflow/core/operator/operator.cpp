#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

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

}  // namespace

void Operator::InitFromOpConf(const OperatorConf& op_conf) {
  OperatorConf* this_op_conf = op_attribute_.mutable_op_conf();
  *this_op_conf = op_conf;
  if (GlobalJobDesc().IsPredict()) { this_op_conf->set_trainable(false); }
  if (this_op_conf->has_enable_cudnn() == false) {
    this_op_conf->set_enable_cudnn(GlobalJobDesc().EnableCudnn());
  }
  if (GetActivationType() != ActivationType::kNone) { EnrollBwBufBn("bw_activation"); }
  InitFromOpConf();
}

LogicalNode* Operator::NewProperLogicalNode() const { return new NormalForwardLogicalNode; }

const LogicalBlobId& Operator::BnInOp2Lbi(const std::string& bn_in_op) const {
  return op_attribute_.bn_in_op2lbi().at(bn_in_op);
}

LogicalBlobId* Operator::MutBnInOp2Lbi(const std::string& bn_in_op) {
  auto it = op_attribute_.mutable_bn_in_op2lbi()->find(bn_in_op);
  if (it == op_attribute_.mutable_bn_in_op2lbi()->end()) {
    return nullptr;
  } else {
    return &(it->second);
  }
}

const std::string& Operator::SoleIbn() const {
  CHECK_EQ(input_bns().size(), 1);
  return input_bns().Get(0);
}
const std::string& Operator::SoleIdbn() const {
  CHECK_EQ(input_diff_bns().size(), 1);
  return input_diff_bns().Get(0);
}
const std::string& Operator::SoleObn() const {
  CHECK_EQ(output_bns().size(), 1);
  return output_bns().Get(0);
}
const std::string& Operator::SoleOdbn() const {
  CHECK_EQ(output_diff_bns().size(), 1);
  return output_diff_bns().Get(0);
}
const std::string& Operator::SoleDtbn() const {
  CHECK_EQ(data_tmp_bns().size(), 1);
  return data_tmp_bns().Get(0);
}
const std::string& Operator::SoleFbbn() const {
  CHECK_EQ(fw_buf_bns().size(), 1);
  return fw_buf_bns().Get(0);
}
const std::string& Operator::SoleBbbn() const {
  CHECK_EQ(bw_buf_bns().size(), 1);
  return bw_buf_bns().Get(0);
}

void Operator::InferBlobDescsIf(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx, int64_t record_piece_size,
                                std::function<void(OpContext*)> EnrollOpCtx) const {
  InferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, record_piece_size, EnrollOpCtx);
  if (op_attribute_.model_bns().size() > 0) {
    InferTotalInstanceNumDesc(GetBlobDesc4BnInOp, parallel_ctx, EnrollOpCtx);
  }
}

void Operator::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx, int64_t record_piece_size,
                              std::function<void(OpContext*)> EnrollOpCtx) const {
  InferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, record_piece_size);
}

void Operator::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx,
                              int64_t record_piece_size) const {
  InferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

void Operator::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  UNIMPLEMENTED() << typeid(*this).name();
}

void Operator::InferBwBufBlobDescsIf(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const OpContext* op_ctx) const {
  InferBwBufBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, op_ctx);
  if (GetActivationType() != ActivationType::kNone) {
    *GetBlobDesc4BnInOp("bw_activation") = *GetBlobDesc4BnInOp(SoleOdbn());
  }
}

void Operator::InferOutputBlobTimeShapeIf(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  for (const std::string& ibn : input_bns()) {
    CHECK_EQ(GetTimeShape4BnInOp(ibn)->elem_cnt(),
             GetTimeShape4BnInOp(input_bns().Get(0))->elem_cnt());
  }
  InferOutputBlobTimeShape(GetTimeShape4BnInOp, parallel_ctx, time_shape);
}

void Operator::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp, const ParallelContext*,
    Shape* time_shape) const {
  if (input_bns().empty() == false) {
    *time_shape = *GetTimeShape4BnInOp(input_bns().Get(0));
  } else {
    *time_shape = Shape({GlobalJobDesc().TotalBatchNum(), GlobalJobDesc().NumOfPiecesInBatch()});
  }
}

void Operator::GetSbpSignaturesIf(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  GetSbpSignatures(LogicalBlobDesc4Ibn, sbp_sig_list);
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Broadcast(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

void Operator::InferSbpSignatureIf(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  if (parallel_desc.parallel_num() == 1) {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    for (const auto& ibn : input_bns()) { (*bn2sbp)[ibn].mutable_split_parallel()->set_axis(0); }
    for (const auto& obn : output_bns()) { (*bn2sbp)[obn].mutable_split_parallel()->set_axis(0); }
  } else if (parallel_desc.parallel_num() > 1) {
    InferSbpSignature(sbp_signature, sbp_sig_conf, CalcOrderValue4SbpSig, SbpInferHint4Ibn,
                      parallel_desc);
  } else {
    UNIMPLEMENTED();
  }
}

void Operator::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  // get op sbp signatures
  auto LogicalBlobDesc4Ibn = [&](const std::string& ibn) -> const BlobDesc& {
    return SbpInferHint4Ibn(ibn).logical_blob_desc();
  };
  SbpSignatureList sbp_sig_list;
  GetSbpSignaturesIf(LogicalBlobDesc4Ibn, &sbp_sig_list);
  // filter sbp signatures by sbp signature conf
  SbpSignatureList filtered_sbp_sigs_by_conf;
  FilterSbpSignatureList(sbp_sig_list, sbp_sig_conf, &filtered_sbp_sigs_by_conf);
  CHECK_GT(filtered_sbp_sigs_by_conf.sbp_signature_size(), 0);
  if (filtered_sbp_sigs_by_conf.sbp_signature_size() == 1) {
    *sbp_signature = *filtered_sbp_sigs_by_conf.sbp_signature().begin();
    return;
  }
  // sort sbp signatures by copy cost, then return the one with least cost
  HashMap<std::string, const SbpParallel*> ibn2producer_sbp_parallel;
  for (const auto& ibn : input_bns()) {
    ibn2producer_sbp_parallel[ibn] = &SbpInferHint4Ibn(ibn).sbp_parallel();
  }
  std::vector<const SbpSignature*> sorted_sbp_signatures;
  SortSbpSignatureListByCopyCost(filtered_sbp_sigs_by_conf, input_bns(), SbpInferHint4Ibn,
                                 CalcOrderValue4SbpSig, &sorted_sbp_signatures);
  *sbp_signature = *sorted_sbp_signatures.at(0);
}

bool Operator::HasOutDiff4Lbi(const LogicalBlobId& lbi) const {
  CHECK_EQ(lbi.op_name(), op_name());
  return std::find_if(
             output_diff_bns().cbegin(), output_diff_bns().cend(),
             [&](const std::string& diff_bn) { return BnInOp2Lbi(GenUnDiffBn(diff_bn)) == lbi; })
         != output_diff_bns().end();
}

void Operator::InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx,
                                   const OpContext* op_ctx) const {
  InferBwBufBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

void Operator::FixInDiffBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* ctx) const {
  VirtualFixInDiffBlobDescs(GetBlobDesc4BnInOp, ctx);
  for (const std::string& input_diff_bn : input_diff_bns()) {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(input_diff_bn);
    if (!blob_desc) { continue; }
    blob_desc->set_has_loss_instance_num_field(true);
  }
}

void Operator::FixParallelDesc(ParallelDesc* pr_desc) const {
  // TODO(): outdated, delete
  VirtualFixParallelDesc(pr_desc);
}

void Operator::FixLbiWhenShareModel(const std::string& shared_op_name) {
  for (const std::string& model_bn : model_bns()) {
    mut_bn_in_op2lbi()->at(model_bn).set_op_name(shared_op_name);
    mut_bn_in_op2lbi()->at(GenDiffBn(model_bn)).set_op_name(shared_op_name);
  }
  for (const std::string& const_model_bn : const_model_bns()) {
    mut_bn_in_op2lbi()->at(const_model_bn).set_op_name(shared_op_name);
  }
}

static bool HasBlobDescWithField(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const PbRpf<std::string>& bn_in_ops, bool (BlobDesc::*has_field)() const) {
  for (const std::string& bn_in_op : bn_in_ops) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn_in_op);
    if (blob_desc && (blob_desc->*has_field)()) { return true; }
  }
  return false;
}

static bool DoAllBlobDescHaveField(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const PbRpf<std::string>& bn_in_ops, bool (BlobDesc::*has_field)() const) {
  for (const std::string& bn_in_op : bn_in_ops) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn_in_op);
    if (blob_desc && !(blob_desc->*has_field)()) { return false; }
  }
  return true;
}

static bool HaveSameDim0InnerShape(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const PbRpf<std::string>& input_bns, const PbRpf<std::string>& output_bns) {
  auto ForEachBn = [&](const std::function<void(const std::string&)>& Handler) {
    for (const auto& bn : input_bns) { Handler(bn); }
    for (const auto& bn : output_bns) { Handler(bn); }
  };
  bool ret = true;
  std::unique_ptr<Shape> dim0_inner_shape;
  ForEachBn([&](const std::string& bn) {
    if (ret == false) { return; }
    const auto& inner_shape = GetBlobDesc4BnInOp(bn)->dim0_inner_shape();
    if (dim0_inner_shape) {
      if (*dim0_inner_shape != inner_shape) { ret = false; }
    } else {
      dim0_inner_shape.reset(new Shape(inner_shape));
    }
  });
  return ret;
}

ActivationType Operator::GetActivationType() const {
  if (HasFieldInCustomizedConf("activation")) {
    return static_cast<ActivationType>(GetEnumFromCustomizedConf("activation"));
  } else {
    return ActivationType::kNone;
  }
}

void Operator::GenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             bool is_forward, const ParallelContext* parallel_ctx,
                             KernelConf* kernel_conf, const OpContext* op_ctx) const {
  *(kernel_conf->mutable_op_attribute()) = op_attribute_;
  if (HasBlobDescWithField(GetBlobDesc4BnInOp, output_bns(), &BlobDesc::header_is_opaque)) {
    kernel_conf->set_need_do_opaque_header(true);
  } else {
    if (HasBlobDescWithField(GetBlobDesc4BnInOp, output_bns(), &BlobDesc::has_data_id_field)) {
      kernel_conf->set_need_do_data_id(true);
    }
    const PbRpf<std::string>* bns = &output_bns();
    if (IsLossOp()) { bns = &input_bns(); }
    if (HasBlobDescWithField(GetBlobDesc4BnInOp, *bns, &BlobDesc::has_col_num_field)) {
      kernel_conf->set_need_do_col_num(true);
    }
    if (HasBlobDescWithField(GetBlobDesc4BnInOp, *bns, &BlobDesc::has_dim0_valid_num_field)) {
      kernel_conf->set_need_do_dim0_valid_num(true);
      if (DoAllBlobDescHaveField(GetBlobDesc4BnInOp, input_bns(),
                                 &BlobDesc::has_dim0_valid_num_field)
          && DoAllBlobDescHaveField(GetBlobDesc4BnInOp, output_bns(),
                                    &BlobDesc::has_dim0_valid_num_field)
          && HaveSameDim0InnerShape(GetBlobDesc4BnInOp, input_bns(), output_bns())) {
        kernel_conf->set_can_naive_do_dim0_valid_num(true);
      }
    }
    if (HasBlobDescWithField(GetBlobDesc4BnInOp, *bns, &BlobDesc::has_dim1_valid_num_field)) {
      kernel_conf->set_need_do_dim1_valid_num(true);
    }
    if (HasBlobDescWithField(GetBlobDesc4BnInOp, *bns, &BlobDesc::has_dim2_valid_num_field)) {
      kernel_conf->set_need_do_dim2_valid_num(true);
    }
    if (HasBlobDescWithField(GetBlobDesc4BnInOp, *bns,
                             &BlobDesc::has_record_id_in_device_piece_field)) {
      kernel_conf->set_need_do_record_id_in_device_piece(true);
      if (DoAllBlobDescHaveField(GetBlobDesc4BnInOp, input_bns(),
                                 &BlobDesc::has_record_id_in_device_piece_field)
          && DoAllBlobDescHaveField(GetBlobDesc4BnInOp, output_bns(),
                                    &BlobDesc::has_record_id_in_device_piece_field)) {
        kernel_conf->set_can_naive_do_record_id_in_device_piece(true);
      }
    }
  }

  kernel_conf->set_is_forward(is_forward);
  DataType data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, output_bns());
  if (data_type == DataType::kInvalidDataType) {
    data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, input_bns());
  }
  if (data_type == DataType::kInvalidDataType) {
    data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, output_diff_bns());
  }
  kernel_conf->set_data_type(data_type);

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
    cudnn_buf_limit_mbyte = GlobalJobDesc().cudnn_buf_limit_mbyte();
  }
  return cudnn_buf_limit_mbyte * 1024 * 1024;
}

std::string Operator::Bn2ConfName(const std::string& bn) const {
  const PbFd* fd = GetCustomizedConf().GetDescriptor()->FindFieldByName(bn);
  if (fd) {
    return GetValFromCustomizedConf<std::string>(bn);
  } else {
    const std::pair<std::string, int32_t> prefix_idx = GenUnRepeatedBn(bn);
    return GetPbRpfFromCustomizedConf<std::string>(prefix_idx.first).Get(prefix_idx.second);
  }
}

LogicalBlobId Operator::ibn2lbi(const std::string& input_bn) const {
  return GenLogicalBlobId(Bn2ConfName(input_bn));
}
LogicalBlobId Operator::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(Bn2ConfName(output_bn));
  return ret;
}
LogicalBlobId Operator::cmbn2lbi(const std::string& const_model_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(const_model_bn);
  return ret;
}
LogicalBlobId Operator::cbbn2lbi(const std::string& const_buf_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(const_buf_bn);
  return ret;
}
LogicalBlobId Operator::mbn2lbi(const std::string& model_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(model_bn);
  return ret;
}
LogicalBlobId Operator::fwmbn2lbi(const std::string& forward_model_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(forward_model_bn);
  return ret;
}

void Operator::EnrollDataTmpBn(const std::string& dtbn) {
  *(mut_data_tmp_bns()->Add()) = dtbn;
  CHECK(mut_bn_in_op2lbi()->insert({dtbn, dtbn2lbi(dtbn)}).second);
}

void Operator::EnrollFwBufBn(const std::string& fbbn) {
  *(mut_fw_buf_bns()->Add()) = fbbn;
  CHECK(mut_bn_in_op2lbi()->insert({fbbn, fbbn2lbi(fbbn)}).second);
}

void Operator::EnrollBwBufBn(const std::string& bbbn) {
  *(mut_bw_buf_bns()->Add()) = bbbn;
  CHECK(mut_bn_in_op2lbi()->insert({bbbn, bbbn2lbi(bbbn)}).second);
}

InputBlobModifier* Operator::EnrollInputBn(const std::string& ibn, bool has_diff) {
  LogicalBlobId lbi = ibn2lbi(ibn);
  CHECK(op_attribute_.mutable_ibn2input_blob_modifier()->insert({ibn, InputBlobModifier()}).second);
  *(mut_input_bns()->Add()) = ibn;
  CHECK(mut_bn_in_op2lbi()->insert({ibn, lbi}).second);
  if (has_diff) {
    std::string idbn = GenDiffBn(ibn);
    *(mut_input_diff_bns()->Add()) = idbn;
    CHECK(mut_bn_in_op2lbi()->insert({idbn, lbi}).second);
  }
  auto* ret = MutInputBlobModifier4Ibn(ibn);
  ret->set_requires_grad(has_diff);
  return ret;
}

const InputBlobModifier& Operator::InputBlobModifier4Ibn(const std::string& ibn) const {
  return op_attribute_.ibn2input_blob_modifier().at(ibn);
}

const OutputBlobModifier& Operator::OutputBlobModifier4Obn(const std::string& obn) const {
  return op_attribute_.obn2output_blob_modifier().at(obn);
}

InputBlobModifier* Operator::MutInputBlobModifier4Ibn(const std::string& ibn) {
  return &op_attribute_.mutable_ibn2input_blob_modifier()->at(ibn);
}

OutputBlobModifier* Operator::MutOutputBlobModifier4Obn(const std::string& obn) {
  return &op_attribute_.mutable_obn2output_blob_modifier()->at(obn);
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
  LogicalBlobId lbi = obn2lbi(obn);
  CHECK(
      op_attribute_.mutable_obn2output_blob_modifier()->insert({obn, OutputBlobModifier()}).second);
  *(mut_output_bns()->Add()) = obn;
  CHECK(mut_bn_in_op2lbi()->insert({obn, lbi}).second);
  if (has_diff) {
    std::string odbn = GenDiffBn(obn);
    *(mut_output_diff_bns()->Add()) = odbn;
    CHECK(mut_bn_in_op2lbi()->insert({odbn, lbi}).second);
  }
  auto* ret = MutOutputBlobModifier4Obn(obn);
  ret->set_requires_grad(has_diff);
  return ret;
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

void Operator::EnrollModelBn(const std::string& mbn) {
  auto Enroll = [&](const std::string& mbn) {
    LogicalBlobId lbi = mbn2lbi(mbn);
    *(mut_model_bns()->Add()) = mbn;
    CHECK(mut_bn_in_op2lbi()->insert({mbn, lbi}).second);
    std::string mdbn = GenDiffBn(mbn);
    *(mut_model_diff_bns()->Add()) = mdbn;
    CHECK(mut_bn_in_op2lbi()->insert({mdbn, lbi}).second);
  };
  Enroll(mbn);
  auto it = op_attribute_.bn_in_op2lbi().find("total_instance_num");
  if (it == op_attribute_.bn_in_op2lbi().end()) { Enroll("total_instance_num"); }
}
void Operator::EnrollModelDiffBn(const std::string& mdbn) {
  LogicalBlobId lbi = mbn2lbi(mdbn);
  *(mut_model_diff_bns()->Add()) = mdbn;
  CHECK(mut_bn_in_op2lbi()->insert({mdbn, lbi}).second);
}
void Operator::EnrollConstModelBn(const std::string& cmbn) {
  *(mut_const_model_bns()->Add()) = cmbn;
  CHECK(mut_bn_in_op2lbi()->insert({cmbn, cmbn2lbi(cmbn)}).second);
}
void Operator::EnrollConstBufBn(const std::string& cbbn) {
  *(mut_const_buf_bns()->Add()) = cbbn;
  CHECK(mut_bn_in_op2lbi()->insert({cbbn, cbbn2lbi(cbbn)}).second);
}
void Operator::EnrollForwardModelBn(const std::string& fwmbn) {
  LogicalBlobId lbi = fwmbn2lbi(fwmbn);
  *(mut_forward_model_bns()->Add()) = fwmbn;
  CHECK(mut_bn_in_op2lbi()->insert({fwmbn, lbi}).second);
}

void Operator::StrFieldTolower(const std::string& field_name) {
  std::string field_val = GetValFromCustomizedConf<std::string>(field_name);
  std::transform(field_val.begin(), field_val.end(), field_val.begin(), ::tolower);
  SetValInCustomizedConf(field_name, field_val);
}

LogicalBlobId Operator::dtbn2lbi(const std::string& data_tmp_bn) const {
  LogicalBlobId lbi;
  lbi.set_op_name(op_name());
  lbi.set_blob_name(data_tmp_bn);
  return lbi;
}

void Operator::InferTotalInstanceNumDesc(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, std::function<void(OpContext*)> EnrollOpCtx) const {
  CHECK_GE(op_attribute_.model_bns().size(), 2);
  auto it = op_attribute_.bn_in_op2lbi().find("total_instance_num");
  if (it != op_attribute_.bn_in_op2lbi().end()) {
    GetBlobDesc4BnInOp("total_instance_num")->mut_shape() = Shape({1});
    for (const std::string& bn : op_attribute_.model_bns()) {
      if (bn != "total_instance_num") {
        GetBlobDesc4BnInOp("total_instance_num")
            ->set_data_type(GetBlobDesc4BnInOp(bn)->data_type());
        break;
      }
    }
  }
}

std::string GenDiffBn(const std::string& bn) { return bn + "_diff"; }
std::string GenUnDiffBn(const std::string& diff_bn) {
  CHECK_STREQ(diff_bn.substr(diff_bn.size() - 5).c_str(), "_diff");
  return diff_bn.substr(0, diff_bn.size() - 5);
}

std::string GenRepeatedBn(const std::string& bn_prefix, int32_t idx) {
  CHECK_GE(idx, 0);
  return bn_prefix + "_" + std::to_string(idx);
}

std::pair<std::string, int32_t> GenUnRepeatedBn(const std::string& bn) {
  const size_t underline_pos = bn.rfind('_');
  CHECK_NE(underline_pos, std::string::npos);
  CHECK_GT(underline_pos, 0);
  CHECK_LT(underline_pos, bn.size() - 1);
  const std::string prefix = bn.substr(0, underline_pos);
  const int32_t idx = oneflow_cast<int32_t>(bn.substr(underline_pos + 1));
  CHECK_GE(idx, 0);
  return std::make_pair(prefix, idx);
}

bool IsOpOnlyCpuSupported(OperatorConf::OpTypeCase op_type_case) {
  return *std::unique_ptr<OnlyCpuSupportPredicator>(NewObj<OnlyCpuSupportPredicator>(op_type_case));
}

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf) {
  Operator* rptr = NewObj<Operator>(op_conf.op_type_case(), op_conf);
  if (IsOpOnlyCpuSupported(op_conf.op_type_case())) {
    CHECK_EQ(op_conf.device_type(), DeviceType::kCPU);
  }
  rptr->InitFromOpConf(op_conf);
  return std::shared_ptr<Operator>(rptr);
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

void Operator::NaiveInferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  if (output_bns().empty()) { return; }
  CHECK_GT(input_bns().size(), 0);
  CHECK_EQ(output_bns().size(), 1);
  bool has_batch_dim = false;
  for (const auto& ibn : input_bns()) { has_batch_dim = has_batch_dim || *HasBatchDim4BnInOp(ibn); }
  *HasBatchDim4BnInOp(SoleObn()) = has_batch_dim;
}

}  // namespace oneflow
