#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

DataType GetDataTypeFromBnInOpVec(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const std::vector<std::string>& bn_in_ops) {
  for (const std::string& bn_in_op : bn_in_ops) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn_in_op);
    if (blob_desc) { return blob_desc->data_type(); }
  }
  return DataType::kInvalidDataType;
}

}  // namespace

void Operator::InitFromOpConf(const OperatorConf& op_conf) {
  op_conf_ = op_conf;
  if (op_conf_.has_use_cudnn_on_gpu() == false) {
    op_conf_.set_use_cudnn_on_gpu(Global<JobDesc>::Get()->UseCudnnOnGpu());
  }
  if (HasFieldInCustomizedConf("activation")) {
    ActivationType activation =
        static_cast<ActivationType>(GetEnumFromCustomizedConf("activation"));
    if (activation != ActivationType::kNone) {
      EnrollDataTmpBn("activation_buf");
    }
  }
  InitFromOpConf();
}

const std::string& Operator::Lbn4BnInOp(const std::string& bn_in_op) const {
  return bn_in_op2lbn_.at(bn_in_op);
}

int8_t Operator::TryModifyLbn4BnInOp(const std::string& bn_in_op,
                                     const std::string& lbn) {
  auto it = bn_in_op2lbn_.find(bn_in_op);
  if (it == bn_in_op2lbn_.end()) { return -1; }
  it->second = lbn;
  return 0;
}

void Operator::ModifyLbn4BnInOp(const std::string& bn_in_op,
                                const std::string& lbn) {
  CHECK_EQ(TryModifyLbn4BnInOp(bn_in_op, lbn), 0);
}

bool Operator::UseCudnn(DeviceType device_type) const {
  return device_type == DeviceType::kGPU && op_conf_.use_cudnn_on_gpu();
}

const std::string& Operator::SoleIbn() const {
  CHECK_EQ(input_bns_.size(), 1);
  return *(input_bns_.begin());
}
const std::string& Operator::SoleIdbn() const {
  CHECK_EQ(input_diff_bns_.size(), 1);
  return *(input_diff_bns_.begin());
}
const std::string& Operator::SoleObn() const {
  CHECK_EQ(output_bns_.size(), 1);
  return *(output_bns_.begin());
}
const std::string& Operator::SoleOdbn() const {
  CHECK_EQ(output_diff_bns_.size(), 1);
  return *(output_diff_bns_.begin());
}
const std::string& Operator::SoleDtbn() const {
  CHECK_EQ(data_tmp_bns_.size(), 1);
  return *(data_tmp_bns_.begin());
}

void Operator::InferBlobDescsIf(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, DeviceType device_type,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  InferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, device_type, EnrollOpCtx);
  if (HasFieldInCustomizedConf("activation")) {
    ActivationType activation =
        static_cast<ActivationType>(GetEnumFromCustomizedConf("activation"));
    if (activation != ActivationType::kNone
        && Global<JobDesc>::Get()->IsTrain()) {
      BlobDesc* buf_blob_desc = GetBlobDesc4BnInOp("activation_buf");
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(SoleObn());
      *buf_blob_desc = *out_blob_desc;
    }
  }
}

void Operator::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, DeviceType device_type,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  InferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, device_type);
}

void Operator::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, DeviceType device_type) const {
  InferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}
void Operator::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  UNIMPLEMENTED() << typeid(*this).name();
}

void Operator::FixParallelDesc(ParallelDesc* pr_desc) const {
  if (IsDecodeOp()) {
    CHECK_EQ(pr_desc->parallel_num(),
             Global<JobDesc>::Get()->job_conf().data_part_num())
        << "parallel_num of data loader is not equal to the data_part_num in "
           "job.prototxt";
  }
  if (model_bns_.empty()) {
    CHECK(model_tmp_bns_.empty());
    pr_desc->set_policy(ParallelPolicy::kDataParallel);
  }
  if (pr_desc->policy() == kModelParallel && MaxModelSplitNum() != -1) {
    pr_desc->RemoveNeedlessDevice(op_name(), MaxModelSplitNum());
  }
  if (pr_desc->policy() == kDataParallel) {
    pr_desc->RemoveNeedlessDevice(op_name(),
                                  Global<JobDesc>::Get()->ParallelPieceSize());
  }
  VirtualFixParallelDesc(pr_desc);
}

void Operator::FixLbnWhenShareModel(const std::string& shared_op_name) {
  for (const std::string& model_bn : model_bns_) {
    std::string model_lbn = shared_op_name + "/" + model_bn;
    bn_in_op2lbn_.at(model_bn) = model_lbn;
    bn_in_op2lbn_.at(GenDiffBn(model_bn)) = model_lbn;
  }
  for (const std::string& model_tmp_bn : model_tmp_bns_) {
    std::string model_tmp_lbn = shared_op_name + "/" + model_tmp_bn;
    bn_in_op2lbn_.at(model_tmp_bn) = model_tmp_lbn;
  }
}

static bool HasBlobDescWithField(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const std::vector<std::string>& bn_in_ops,
    bool (BlobDesc::*has_field)() const) {
  for (const std::string& bn_in_op : bn_in_ops) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn_in_op);
    if (blob_desc && (blob_desc->*has_field)()) { return true; }
  }
  return false;
}

void Operator::GenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    bool is_forward, DeviceType device_type,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf,
    const OpContext* op_ctx) const {
  *(kernel_conf->mutable_op_conf()) = op_conf_;
  *(kernel_conf->mutable_bn_in_op2lbn()) = HashMap2PbMap(bn_in_op2lbn_);
  *(kernel_conf->mutable_data_tmp_bns()) = StdVec2PbRpf(data_tmp_bns_);
  *(kernel_conf->mutable_input_bns()) = StdVec2PbRpf(input_bns_);
  *(kernel_conf->mutable_input_diff_bns()) = StdVec2PbRpf(input_diff_bns_);
  *(kernel_conf->mutable_output_bns()) = StdVec2PbRpf(output_bns_);
  *(kernel_conf->mutable_output_diff_bns()) = StdVec2PbRpf(output_diff_bns_);
  *(kernel_conf->mutable_model_bns()) = StdVec2PbRpf(model_bns_);
  *(kernel_conf->mutable_model_diff_bns()) = StdVec2PbRpf(model_diff_bns_);
  *(kernel_conf->mutable_model_tmp_bns()) = StdVec2PbRpf(model_tmp_bns_);
  *(kernel_conf->mutable_forward_model_bns()) =
      StdVec2PbRpf(forward_model_bns_);
  kernel_conf->set_need_do_data_id(false);
  if (HasBlobDescWithField(GetBlobDesc4BnInOp, output_bns_,
                           &BlobDesc::has_data_id_field)) {
    kernel_conf->set_need_do_data_id(true);
  }
  kernel_conf->set_need_do_col_num(false);
  const std::vector<std::string>* bns = &output_bns_;
  if (IsLossOp()) { bns = &input_bns_; }
  if (HasBlobDescWithField(GetBlobDesc4BnInOp, *bns,
                           &BlobDesc::has_col_num_field)) {
    kernel_conf->set_need_do_col_num(true);
  }

  kernel_conf->set_is_forward(is_forward);
  DataType data_type =
      GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, output_bns_);
  if (data_type == DataType::kInvalidDataType) {
    data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, input_bns_);
  }
  if (IsCloneOp()) {
    data_type = GetDataTypeFromBnInOpVec(GetBlobDesc4BnInOp, output_diff_bns_);
  }
  kernel_conf->set_data_type(data_type);
  kernel_conf->set_device_type(device_type);
  VirtualGenKernelConf(GetBlobDesc4BnInOp, parallel_ctx, kernel_conf, op_ctx);
}

void Operator::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf,
    const OpContext* op_ctx) const {
  VirtualGenKernelConf(GetBlobDesc4BnInOp, parallel_ctx, kernel_conf);
}

std::string Operator::ibn2lbn(const std::string& input_bn) const {
  const google::protobuf::Descriptor* desc =
      GetCustomizedConf().GetDescriptor();
  const google::protobuf::FieldDescriptor* fd = desc->FindFieldByName(input_bn);
  if (fd) {
    return GetValFromCustomizedConf<std::string>(input_bn);
  } else {
    size_t underline_pos = input_bn.rfind('_');
    CHECK_NE(underline_pos, std::string::npos);
    std::string ibn_prefix = input_bn.substr(0, underline_pos);
    int32_t ibn_idx = oneflow_cast<int32_t>(input_bn.substr(underline_pos + 1));
    return GetPbRpfFromCustomizedConf<std::string>(ibn_prefix).Get(ibn_idx);
  }
}
std::string Operator::obn2lbn(const std::string& output_bn) const {
  return op_name() + "/" + GetValFromCustomizedConf<std::string>(output_bn);
}
std::string Operator::mtbn2lbn(const std::string& model_tmp_bn) const {
  return op_name() + "/" + model_tmp_bn;
}
std::string Operator::mbn2lbn(const std::string& model_bn) const {
  return op_name() + "/" + model_bn;
}
std::string Operator::fwmbn2lbn(const std::string& forward_model_bn) const {
  return op_name() + "/" + forward_model_bn;
}

void Operator::EnrollDataTmpBn(const std::string& dtbn) {
  data_tmp_bns_.push_back(dtbn);
  CHECK(bn_in_op2lbn_.emplace(dtbn, dtbn2lbn(dtbn)).second);
}
void Operator::EnrollInputBn(const std::string& ibn, bool has_diff) {
  std::string lbn = ibn2lbn(ibn);
  input_bns_.push_back(ibn);
  CHECK(bn_in_op2lbn_.emplace(ibn, lbn).second);
  if (has_diff) {
    std::string idbn = GenDiffBn(ibn);
    input_diff_bns_.push_back(idbn);
    CHECK(bn_in_op2lbn_.emplace(idbn, lbn).second);
  }
}

void Operator::EnrollRepeatedInputBn(const std::string& ibn_prefix, int32_t num,
                                     bool has_diff) {
  FOR_RANGE(int32_t, i, 0, num) {
    std::string ibn = ibn_prefix + "_" + std::to_string(i);
    EnrollInputBn(ibn, has_diff);
  }
}

void Operator::EnrollRepeatedInputBn(const std::string& ibn_prefix,
                                     bool has_diff) {
  EnrollRepeatedInputBn(
      ibn_prefix, GetPbRpfFromCustomizedConf<std::string>(ibn_prefix).size(),
      has_diff);
}

void Operator::EnrollRepeatedInputBn(const std::string& ibn_prefix,
                                     int32_t num) {
  EnrollRepeatedInputBn(ibn_prefix, num, true);
}

void Operator::EnrollRepeatedInputBn(const std::string& ibn_prefix) {
  EnrollRepeatedInputBn(ibn_prefix, true);
}

void Operator::EnrollOutputBn(const std::string& obn, bool has_diff) {
  std::string lbn = obn2lbn(obn);
  output_bns_.push_back(obn);
  CHECK(bn_in_op2lbn_.emplace(obn, lbn).second);
  if (has_diff) {
    std::string odbn = GenDiffBn(obn);
    output_diff_bns_.push_back(odbn);
    CHECK(bn_in_op2lbn_.emplace(odbn, lbn).second);
  }
}
void Operator::EnrollModelBn(const std::string& mbn) {
  if (op_conf_.trainable() == false) {
    EnrollModelTmpBn(mbn);
    return;
  }
  std::string lbn = mbn2lbn(mbn);
  model_bns_.push_back(mbn);
  CHECK(bn_in_op2lbn_.emplace(mbn, lbn).second);
  std::string mdbn = GenDiffBn(mbn);
  model_diff_bns_.push_back(mdbn);
  CHECK(bn_in_op2lbn_.emplace(mdbn, lbn).second);
}
void Operator::EnrollModelTmpBn(const std::string& mtbn) {
  model_tmp_bns_.push_back(mtbn);
  CHECK(bn_in_op2lbn_.emplace(mtbn, mtbn2lbn(mtbn)).second);
}
void Operator::EnrollForwardModelBn(const std::string& fwmbn) {
  std::string lbn = fwmbn2lbn(fwmbn);
  forward_model_bns_.push_back(fwmbn);
  CHECK(bn_in_op2lbn_.emplace(fwmbn, lbn).second);
}

void Operator::StrFieldTolower(const std::string& field_name) {
  std::string field_val = GetValFromCustomizedConf<std::string>(field_name);
  std::transform(field_val.begin(), field_val.end(), field_val.begin(),
                 ::tolower);
  SetValInCustomizedConf(field_name, field_val);
}

std::string Operator::dtbn2lbn(const std::string& data_tmp_bn) const {
  return op_name() + "/" + data_tmp_bn;
}

std::string GenDiffBn(const std::string& bn) { return bn + "_diff"; }
std::string GenUnDiffBn(const std::string& diff_bn) {
  CHECK_STREQ(diff_bn.substr(diff_bn.size() - 5).c_str(), "_diff");
  return diff_bn.substr(0, diff_bn.size() - 5);
}
std::string GenUnCloneLbn(const std::string& clone_lbn) {
  CHECK_STREQ(clone_lbn.substr(0, 6).c_str(), "clone_");
  int32_t before_num = clone_lbn.size() - 1;
  while (std::isdigit(clone_lbn.at(before_num))) { --before_num; }
  CHECK_STREQ(clone_lbn.substr(before_num - 4, 5).c_str(), "/out_");
  return clone_lbn.substr(6, before_num - 10);
}
std::string GetOpNameFromLbn(const std::string& lbn) {
  return ParseLbn(lbn).first;
}
std::string GetBnInOpFromLbn(const std::string& lbn) {
  return ParseLbn(lbn).second;
}
std::pair<std::string, std::string> ParseLbn(const std::string& lbn) {
  size_t pos = lbn.find('/');
  CHECK_NE(pos, std::string::npos);
  return {lbn.substr(0, pos), lbn.substr(pos + 1)};
}

static HashMap<int, std::function<Operator*(const OperatorConf&)>>&
OpTypeCase2Creator() {
  static HashMap<int, std::function<Operator*(const OperatorConf&)>> obj;
  return obj;
}

void AddOpCreator(OperatorConf::OpTypeCase op_type_case,
                  std::function<Operator*(const OperatorConf&)> creator) {
  CHECK(OpTypeCase2Creator().emplace(op_type_case, creator).second);
}

void AddOpCreator(OperatorConf::OpTypeCase op_type_case,
                  std::function<Operator*()> creator) {
  CHECK(OpTypeCase2Creator()
            .emplace(op_type_case,
                     [creator](const OperatorConf&) { return creator(); })
            .second);
}

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf) {
  Operator* rptr = OpTypeCase2Creator().at(op_conf.op_type_case())(op_conf);
  std::shared_ptr<Operator> ret(rptr);
  ret->InitFromOpConf(op_conf);
  return ret;
}

void EraseEmptyBnInVec(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
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
