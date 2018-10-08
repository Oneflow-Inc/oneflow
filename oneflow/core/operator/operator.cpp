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

bool IsOpWithModel(const OpAttribute& op_attribute) {
  if (op_attribute.model_bns().empty() == false) { return true; }
  if (op_attribute.const_model_bns().empty() == false) { return true; }
  if (op_attribute.forward_model_bns().empty() == false) { return true; }
  return false;
}

}  // namespace

void Operator::InitFromOpConf(const OperatorConf& op_conf) {
  OperatorConf* this_op_conf = op_attribute_.mutable_op_conf();
  *this_op_conf = op_conf;
  if (Global<JobDesc>::Get()->IsPredict()) { this_op_conf->set_trainable(false); }
  if (this_op_conf->has_enable_cudnn() == false) {
    this_op_conf->set_enable_cudnn(Global<JobDesc>::Get()->EnableCudnn());
  }
  std::string md_load_dir = JoinPath(Global<JobDesc>::Get()->ModelLoadPath(), op_conf.name());
  if (SnapshotFS()->FileExists(md_load_dir) && SnapshotFS()->IsDirectory(md_load_dir)) {
    this_op_conf->set_model_load_dir(md_load_dir);
  }
  if (GetActivationType() != ActivationType::kNone) { EnrollBwBufBn("bw_activation"); }
  InitFromOpConf();
  if (IsOpWithModel(op_attribute_) && this_op_conf->trainable() == false) {
    CHECK(this_op_conf->has_model_load_dir());
  }
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
const std::string& Operator::SolePibn() const {
  CHECK_EQ(pb_input_bns().size(), 1);
  return pb_input_bns().Get(0);
}
const std::string& Operator::SoleIdbn() const {
  CHECK_EQ(input_diff_bns().size(), 1);
  return input_diff_bns().Get(0);
}
const std::string& Operator::SoleObn() const {
  CHECK_EQ(output_bns().size(), 1);
  return output_bns().Get(0);
}
const std::string& Operator::SolePobn() const {
  CHECK_EQ(pb_output_bns().size(), 1);
  return pb_output_bns().Get(0);
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
std::string Operator::RepeatedIbn(const std::string& prefix, int32_t idx) const {
  CHECK_LT(idx, RepeatedIbnSize(prefix));
  return prefix + "_" + std::to_string(idx);
}
int32_t Operator::RepeatedIbnSize(const std::string& prefix) const {
  int32_t ret = 0;
  ForEachInputBn([&ret, &prefix](const std::string& ibn) {
    std::string idx_str = std::to_string(ret);
    size_t idx_size = idx_str.size();
    size_t ibn_size = ibn.size();
    std::string prefix_substr = ibn.substr(0, ibn_size - idx_size - 1);
    if (prefix_substr == prefix) { ret++; }
  });
  return ret;
}
std::string Operator::RepeatedObn(const std::string& prefix, int32_t idx) const {
  CHECK_LT(idx, RepeatedObnSize(prefix));
  return prefix + "_" + std::to_string(idx);
}
int32_t Operator::RepeatedObnSize(const std::string& prefix) const {
  int32_t ret = 0;
  ForEachOutputBn([&ret, &prefix](const std::string& obn) {
    std::string idx_str = std::to_string(ret);
    size_t idx_size = idx_str.size();
    size_t obn_size = obn.size();
    std::string prefix_substr = obn.substr(0, obn_size - idx_size - 1);
    if (prefix_substr == prefix) { ret++; }
  });
  return ret;
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

static bool HasAllBlobDescWithField(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const PbRpf<std::string>& bn_in_ops, bool (BlobDesc::*has_field)() const) {
  for (const std::string& bn_in_op : bn_in_ops) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn_in_op);
    if (blob_desc && !(blob_desc->*has_field)()) { return false; }
  }
  return true;
}

static bool HasSameInstanceInnerShape(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const PbRpf<std::string>& input_bns, const PbRpf<std::string>& output_bns) {
  auto ForEachBn = [&](const std::function<void(const std::string&)>& Handler) {
    for (const auto& bn : input_bns) { Handler(bn); }
    for (const auto& bn : output_bns) { Handler(bn); }
  };
  bool ret = true;
  std::unique_ptr<Shape> instance_inner_shape;
  ForEachBn([&](const std::string& bn) {
    if (ret == false) { return; }
    const auto& inner_shape = GetBlobDesc4BnInOp(bn)->instance_inner_shape();
    if (instance_inner_shape) {
      if (!(*instance_inner_shape == inner_shape)) { ret = false; }
    } else {
      instance_inner_shape.reset(new Shape(inner_shape));
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
  auto HasBnWithField = [&](const PbRpf<std::string>& bns, bool (BlobDesc::*has_field)() const) {
    return HasBlobDescWithField(GetBlobDesc4BnInOp, bns, has_field);
  };
  *(kernel_conf->mutable_op_attribute()) = op_attribute_;
  CHECK(!HasBnWithField(pb_output_bns(), &BlobDesc::header_is_opaque));
  if (HasBnWithField(output_bns(), &BlobDesc::header_is_opaque)) {
    kernel_conf->set_need_do_opaque_header(true);
  } else {
    if (HasBnWithField(output_bns(), &BlobDesc::has_data_id_field)
        || HasBnWithField(pb_output_bns(), &BlobDesc::has_data_id_field)) {
      kernel_conf->set_need_do_data_id(true);
    }
    const PbRpf<std::string>& obns = IsLossOp() ? input_bns() : output_bns();
    const PbRpf<std::string>& pobns = IsLossOp() ? pb_input_bns() : pb_output_bns();
    if (HasBnWithField(obns, &BlobDesc::has_col_num_field)
        || HasBnWithField(pobns, &BlobDesc::has_col_num_field)) {
      kernel_conf->set_need_do_col_num(true);
    }
    if (HasBlobDescWithField(GetBlobDesc4BnInOp, obns,
                             &BlobDesc::has_instance_varying_elem_cnt_field)) {
      kernel_conf->set_need_do_instance_varying_elem_cnt(true);
      if (HasAllBlobDescWithField(GetBlobDesc4BnInOp, input_bns(),
                                  &BlobDesc::has_instance_varying_elem_cnt_field)
          && HasAllBlobDescWithField(GetBlobDesc4BnInOp, output_bns(),
                                     &BlobDesc::has_instance_varying_elem_cnt_field)) {
        kernel_conf->set_can_naive_do_instance_varying_elem_cnt(true);
      }
    }
    if (HasBlobDescWithField(GetBlobDesc4BnInOp, obns, &BlobDesc::has_varying_instance_num_field)) {
      kernel_conf->set_need_do_varying_instance_num(true);
      if (HasAllBlobDescWithField(GetBlobDesc4BnInOp, input_bns(),
                                  &BlobDesc::has_varying_instance_num_field)
          && HasAllBlobDescWithField(GetBlobDesc4BnInOp, output_bns(),
                                     &BlobDesc::has_varying_instance_num_field)
          && HasSameInstanceInnerShape(GetBlobDesc4BnInOp, input_bns(), output_bns())) {
        kernel_conf->set_can_naive_do_varying_instance_num(true);
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
  if (data_type == DataType::kInvalidDataType) {
    data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, pb_input_bns());
  }
  if (data_type == DataType::kInvalidDataType) {
    data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, pb_output_bns());
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
    cudnn_buf_limit_mbyte = Global<JobDesc>::Get()->cudnn_buf_limit_mbyte();
  }
  return cudnn_buf_limit_mbyte * 1024 * 1024;
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
LogicalBlobId Operator::pibn2lbi(const std::string& pb_input_bn) const {
  LogicalBlobId lbi = ibn2lbi(pb_input_bn);
  lbi.set_is_pb_blob(true);
  return lbi;
}
LogicalBlobId Operator::obn2lbi(const std::string& output_bn) const {
  const google::protobuf::Descriptor* desc = GetCustomizedConf().GetDescriptor();
  const google::protobuf::FieldDescriptor* fd = desc->FindFieldByName(output_bn);
  std::string name;
  if (fd) {
    name = GetValFromCustomizedConf<std::string>(output_bn);
  } else {
    size_t underline_pos = output_bn.rfind('_');
    CHECK_NE(underline_pos, std::string::npos);
    std::string obn_prefix = output_bn.substr(0, underline_pos);
    int32_t obn_idx = oneflow_cast<int32_t>(output_bn.substr(underline_pos + 1));
    name = GetPbRpfFromCustomizedConf<std::string>(obn_prefix).Get(obn_idx);
  }
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(name);
  return ret;
}
LogicalBlobId Operator::pobn2lbi(const std::string& pb_output_bn) const {
  LogicalBlobId lbi = obn2lbi(pb_output_bn);
  lbi.set_is_pb_blob(true);
  return lbi;
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

void Operator::EnrollPbInputBn(const std::string& pibn) {
  LogicalBlobId lbi = pibn2lbi(pibn);
  CHECK(lbi.is_pb_blob());
  *(mut_pb_input_bns()->Add()) = pibn;
  CHECK(mut_bn_in_op2lbi()->insert({pibn, lbi}).second);
}

void Operator::EnrollPbOutputBn(const std::string& pobn) {
  LogicalBlobId lbi = pobn2lbi(pobn);
  CHECK(lbi.is_pb_blob());
  *(mut_pb_output_bns()->Add()) = pobn;
  CHECK(mut_bn_in_op2lbi()->insert({pobn, lbi}).second);
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

void Operator::EnrollRepeatedOutputBn(const std::string& ibn_prefix, int32_t num, bool has_diff) {
  FOR_RANGE(int32_t, i, 0, num) {
    std::string ibn = ibn_prefix + "_" + std::to_string(i);
    EnrollOutputBn(ibn, has_diff);
  }
}

void Operator::EnrollRepeatedOutputBn(const std::string& ibn_prefix, bool has_diff) {
  EnrollRepeatedOutputBn(ibn_prefix, GetPbRpfFromCustomizedConf<std::string>(ibn_prefix).size(),
                         has_diff);
}

void Operator::EnrollRepeatedOutputBn(const std::string& ibn_prefix, int32_t num) {
  EnrollRepeatedOutputBn(ibn_prefix, num, true);
}

void Operator::EnrollRepeatedOutputBn(const std::string& ibn_prefix) {
  EnrollRepeatedOutputBn(ibn_prefix, true);
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

int32_t Operator::GetRepeatedInputBnNum(const std::string& ibn_prefix) const {
  int32_t count = 0;
  for (size_t i = 0; i < input_bns().size(); ++i) {
    if (input_bns().Get(i).compare(0, ibn_prefix.length(), ibn_prefix) == 0) { count++; }
  }
  return count;
}

std::string Operator::GetRepeatedInputBn(const std::string& ibn_prefix, size_t idx) const {
  std::string ibn = ibn_prefix + "_" + std::to_string(idx);
  return ibn;
}

void Operator::ForEachInputBn(const std::function<void(const std::string&)>& Handler) const {
  for (const std::string& ibn : input_bns()) { Handler(ibn); }
  for (const std::string& pibn : pb_input_bns()) { Handler(pibn); }
}

void Operator::ForEachOutputBn(const std::function<void(const std::string&)>& Handler) const {
  for (const std::string& obn : output_bns()) { Handler(obn); }
  for (const std::string& pobn : pb_output_bns()) { Handler(pobn); }
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
