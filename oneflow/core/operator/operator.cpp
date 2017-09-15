#include "oneflow/core/operator/operator.h"

namespace oneflow {

void Operator::InitFromOpConf(const OperatorConf& op_conf) {
  op_conf_ = op_conf;
  InitFromOpConf();
}

void Operator::InitFromProto(const OperatorProto& op_proto) {
  op_conf_ = op_proto.op_conf();
  bn_in_op2lbn_ = PbMap2HashMap(op_proto.bn_in_op2lbn());
  data_tmp_bns_ = PbRpf2StdVec(op_proto.data_tmp_bn());
  input_bns_ = PbRpf2StdVec(op_proto.input_bn());
  input_diff_bns_ = PbRpf2StdVec(op_proto.input_diff_bn());
  output_bns_ = PbRpf2StdVec(op_proto.output_bn());
  output_diff_bns_ = PbRpf2StdVec(op_proto.output_diff_bn());
  model_bns_ = PbRpf2StdVec(op_proto.model_bn());
  model_diff_bns_ = PbRpf2StdVec(op_proto.model_diff_bn());
  model_tmp_bns_ = PbRpf2StdVec(op_proto.model_tmp_bn());
}

void Operator::ToProto(OperatorProto* ret) const {
  *(ret->mutable_op_conf()) = op_conf_;
  *(ret->mutable_bn_in_op2lbn()) = HashMap2PbMap(bn_in_op2lbn_);
  *(ret->mutable_data_tmp_bn()) = StdVec2PbRpf(data_tmp_bns_);
  *(ret->mutable_input_bn()) = StdVec2PbRpf(input_bns_);
  *(ret->mutable_input_diff_bn()) = StdVec2PbRpf(input_diff_bns_);
  *(ret->mutable_output_bn()) = StdVec2PbRpf(output_bns_);
  *(ret->mutable_output_diff_bn()) = StdVec2PbRpf(output_diff_bns_);
  *(ret->mutable_model_bn()) = StdVec2PbRpf(model_bns_);
  *(ret->mutable_model_diff_bn()) = StdVec2PbRpf(model_diff_bns_);
  *(ret->mutable_model_tmp_bn()) = StdVec2PbRpf(model_tmp_bns_);
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

std::string UserOperator::ibn2lbn(const std::string& input_bn) const {
  return GetStringFromSpecialConf(input_bn);
}

std::string UserOperator::obn2lbn(const std::string& output_bn) const {
  return op_name() + "/" + GetStringFromSpecialConf(output_bn);
}

std::string UserOperator::mtbn2lbn(const std::string& model_tmp_bn) const {
  return op_name() + "/" + model_tmp_bn;
}

std::string UserOperator::mbn2lbn(const std::string& model_bn) const {
  return op_name() + "/" + model_bn;
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

}  // namespace oneflow
