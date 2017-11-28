#include "oneflow/core/operator/operator.h"

namespace oneflow {

void Operator::InitFromOpConf(const OperatorConf& op_conf) {
  op_conf_ = op_conf;
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

void Operator::FixParallelDesc(ParallelDesc* pr_desc) const {
  if (model_bns_.empty() && model_tmp_bns_.empty()) {
    pr_desc->set_policy(ParallelPolicy::kDataParallel);
  }
  if (IsDataLoaderOp() == false && IsPrintOp() == false) {
    pr_desc->RemoveInvalidDevice();
  }
  if (pr_desc->policy() == kModelParallel && MaxModelSplitNum() != -1) {
    pr_desc->RemoveNeedlessDevice(MaxModelSplitNum());
  }
  if (pr_desc->policy() == kDataParallel) {
    pr_desc->RemoveNeedlessDevice(JobDesc::Singleton()->ParallelPieceSize());
  }
  VirtualFixParallelDesc(pr_desc);
}

static bool HasBlobDescWithDataId(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const std::vector<std::string>& bn_in_ops) {
  for (const std::string& bn_in_op : bn_in_ops) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn_in_op);
    if (blob_desc && blob_desc->has_data_id()) { return true; }
  }
  return false;
}

void Operator::GenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    bool is_forward, const ParallelContext* parallel_ctx,
    KernelConf* kernel_conf) const {
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
  kernel_conf->set_need_do_data_id(false);
  if (HasBlobDescWithDataId(GetBlobDesc4BnInOp, output_bns_)) {
    kernel_conf->set_need_do_data_id(true);
  }
  kernel_conf->set_is_forward(is_forward);
  if (output_bns_.empty() == false) {
    kernel_conf->set_data_type(GetBlobDesc4BnInOp(output_bns_[0])->data_type());
  } else if (input_bns_.empty() == false) {
    kernel_conf->set_data_type(GetBlobDesc4BnInOp(input_bns_[0])->data_type());
  } else {
    kernel_conf->set_data_type(DataType::kInvalidDataType);
  }
  VirtualGenKernelConf(GetBlobDesc4BnInOp, parallel_ctx, kernel_conf);
}

std::string Operator::ibn2lbn(const std::string& input_bn) const {
  return GetStringFromSpecialConf(input_bn);
}
std::string Operator::obn2lbn(const std::string& output_bn) const {
  return op_name() + "/" + GetStringFromSpecialConf(output_bn);
}
std::string Operator::mtbn2lbn(const std::string& model_tmp_bn) const {
  return op_name() + "/" + model_tmp_bn;
}
std::string Operator::mbn2lbn(const std::string& model_bn) const {
  return op_name() + "/" + model_bn;
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

std::string Operator::dtbn2lbn(const std::string& data_tmp_bn) const {
  return op_name() + "/" + data_tmp_bn;
}

std::string GenDiffBn(const std::string& bn) { return bn + "_diff"; }
std::string GenUnDiffBn(const std::string& diff_bn) {
  CHECK_STREQ(diff_bn.substr(diff_bn.size() - 5).c_str(), "_diff");
  return diff_bn.substr(0, diff_bn.size() - 5);
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

static HashMap<int, std::function<Operator*()>>& OpTypeCase2Creator() {
  static HashMap<int, std::function<Operator*()>> obj;
  return obj;
}

void AddOpCreator(OperatorConf::OpTypeCase op_type_case,
                  std::function<Operator*()> creator) {
  CHECK(OpTypeCase2Creator().emplace(op_type_case, creator).second);
}

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf) {
  Operator* rptr = OpTypeCase2Creator().at(op_conf.op_type_case())();
  std::shared_ptr<Operator> ret(rptr);
  ret->InitFromOpConf(op_conf);
  return ret;
}

}  // namespace oneflow
