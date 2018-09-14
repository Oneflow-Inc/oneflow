#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

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
  if (Global<JobDesc>::Get()->IsPredict()) { this_op_conf->set_trainable(false); }
  if (this_op_conf->has_enable_cudnn() == false) {
    this_op_conf->set_enable_cudnn(Global<JobDesc>::Get()->EnableCudnn());
  }
  if (GetActivationType() != ActivationType::kNone) { EnrollBwBufBn("bw_activation"); }
  InitFromOpConf();
}

LogicalNode* Operator::NewProperLogicalNode() { return new NormalForwardLogicalNode; }

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
                                const ParallelContext* parallel_ctx,
                                std::function<void(OpContext*)> EnrollOpCtx) const {
  InferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, EnrollOpCtx);
}

void Operator::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx,
                              std::function<void(OpContext*)> EnrollOpCtx) const {
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

void Operator::InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx,
                                   const OpContext* op_ctx) const {
  InferBwBufBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

void Operator::FixParallelDesc(ParallelDesc* pr_desc) const {
  if (model_bns().empty() && const_model_bns().empty()) {
    pr_desc->set_policy(ParallelPolicy::kDataParallel);
  }
  if (pr_desc->policy() == kModelParallel && MaxModelSplitNum() != -1) {
    pr_desc->RemoveNeedlessDevice(op_name(), MaxModelSplitNum());
  }
  if (pr_desc->policy() == kDataParallel) {
    pr_desc->RemoveNeedlessDevice(op_name(), Global<JobDesc>::Get()->PieceSize());
  }
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

LogicalBlobId Operator::ibn2lbi(const std::string& input_bn) const {
  const google::protobuf::Descriptor* desc = GetCustomizedConf().GetDescriptor();
  const google::protobuf::FieldDescriptor* fd = desc->FindFieldByName(input_bn);
  std::string name;
  if (fd) {
    name = GetValFromCustomizedConf<std::string>(input_bn);
  } else {
    size_t underline_pos = input_bn.rfind('_');
    CHECK_NE(underline_pos, std::string::npos);
    std::string ibn_prefix = input_bn.substr(0, underline_pos);
    int32_t ibn_idx = oneflow_cast<int32_t>(input_bn.substr(underline_pos + 1));
    name = GetPbRpfFromCustomizedConf<std::string>(ibn_prefix).Get(ibn_idx);
  }
  return GenLogicalBlobId(name);
}
LogicalBlobId Operator::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(GetValFromCustomizedConf<std::string>(output_bn));
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
LogicalBlobId Operator::mmbn2lbi(const std::string& moving_model_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(moving_model_bn);
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

void Operator::EnrollInputBn(const std::string& ibn, bool has_diff) {
  LogicalBlobId lbi = ibn2lbi(ibn);
  *(mut_input_bns()->Add()) = ibn;
  CHECK(mut_bn_in_op2lbi()->insert({ibn, lbi}).second);
  if (has_diff) {
    std::string idbn = GenDiffBn(ibn);
    *(mut_input_diff_bns()->Add()) = idbn;
    CHECK(mut_bn_in_op2lbi()->insert({idbn, lbi}).second);
  }
}

void Operator::EnrollRepeatedInputBn(const std::string& ibn_prefix, int32_t num, bool has_diff) {
  FOR_RANGE(int32_t, i, 0, num) {
    std::string ibn = ibn_prefix + "_" + std::to_string(i);
    EnrollInputBn(ibn, has_diff);
  }
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

void Operator::EnrollOutputBn(const std::string& obn, bool has_diff) {
  LogicalBlobId lbi = obn2lbi(obn);
  *(mut_output_bns()->Add()) = obn;
  CHECK(mut_bn_in_op2lbi()->insert({obn, lbi}).second);
  if (has_diff) {
    std::string odbn = GenDiffBn(obn);
    *(mut_output_diff_bns()->Add()) = odbn;
    CHECK(mut_bn_in_op2lbi()->insert({odbn, lbi}).second);
  }
}
void Operator::EnrollModelBn(const std::string& mbn) {
  if (op_conf().trainable() == false) {
    EnrollConstModelBn(mbn);
    return;
  }
  LogicalBlobId lbi = mbn2lbi(mbn);
  *(mut_model_bns()->Add()) = mbn;
  CHECK(mut_bn_in_op2lbi()->insert({mbn, lbi}).second);
  std::string mdbn = GenDiffBn(mbn);
  *(mut_model_diff_bns()->Add()) = mdbn;
  CHECK(mut_bn_in_op2lbi()->insert({mdbn, lbi}).second);
}
void Operator::EnrollMovingModelBn(const std::string& mmbn) {
  LogicalBlobId lbi = mmbn2lbi(mmbn);
  *(mut_moving_model_bns()->Add()) = mmbn;
  CHECK(mut_bn_in_op2lbi()->insert({mmbn, lbi}).second);
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

std::string GenDiffBn(const std::string& bn) { return bn + "_diff"; }
std::string GenUnDiffBn(const std::string& diff_bn) {
  CHECK_STREQ(diff_bn.substr(diff_bn.size() - 5).c_str(), "_diff");
  return diff_bn.substr(0, diff_bn.size() - 5);
}

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf) {
  Operator* rptr = NewObj<Operator>(op_conf.op_type_case(), op_conf);
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

}  // namespace oneflow
