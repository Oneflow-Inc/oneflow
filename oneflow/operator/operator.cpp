#include "operator/operator.h"

namespace oneflow {
  
void Operator::InitFromOperatorProto(const OperatorProto& op_proto) {
  op_conf_ = op_proto.user_conf();
  
  special_ibn2lbn_ = PbMap2HashMap(op_proto.special_ibn2lbn());
  data_tmp_bns_ = PbVec2StdVec(op_proto.data_tmp_bns());
  input_bns_ = PbVec2StdVec(op_proto.input_bns());
  input_diff_bns_ = PbVec2StdVec(op_proto.input_diff_bns());
  output_bns_ = PbVec2StdVec(op_proto.output_bns());
  output_diff_bns_ = PbVec2StdVec(op_proto.output_diff_bns());
  model_bns_ = PbVec2StdVec(op_proto.model_bns());
  model_diff_bns_ = PbVec2StdVec(op_proto.model_diff_bns());
  model_tmp_bns_ = PbVec2StdVec(op_proto.model_tmp_bns());
}

OperatorProto Operator::ToOperatorProto() {
  OperatorProto op_proto;
  *(op_proto.mutable_user_conf()) = op_conf_;
  *(op_proto.mutable_special_ibn2lbn()) = HashMap2PbMap(special_ibn2lbn_);
  *(op_proto.mutable_data_tmp_bns()) = StdVec2PbVec(data_tmp_bns_);
  *(op_proto.mutable_input_bns()) = StdVec2PbVec(input_bns_);
  *(op_proto.mutable_input_diff_bns()) = StdVec2PbVec(input_diff_bns_);
  *(op_proto.mutable_output_bns()) = StdVec2PbVec(output_bns_);
  *(op_proto.mutable_output_diff_bns()) = StdVec2PbVec(output_diff_bns_);
  *(op_proto.mutable_model_bns()) = StdVec2PbVec(model_bns_);
  *(op_proto.mutable_model_diff_bns()) = StdVec2PbVec(model_diff_bns_);
  *(op_proto.mutable_model_tmp_bns()) = StdVec2PbVec(model_tmp_bns_);
  return op_proto;
}

std::string GenDiffBn(const std::string& bn) {
  return bn + "_diff";
}

std::string GenUnDiffBn(const std::string& diff_bn) {
  CHECK_STREQ(diff_bn.substr(diff_bn.size() - 5).c_str(), "_diff");
  return diff_bn.substr(0, diff_bn.size() - 5);
}

std::string Operator::dtbn2lbn(const std::string& data_tmp_bn) const {
  return op_name() + "/" + data_tmp_bn;
}
std::string Operator::idbn2lbn(const std::string& input_diff_bn) const {
  return ibn2lbn(GenUnDiffBn(input_diff_bn));
}
std::string Operator::odbn2lbn(const std::string& output_diff_bn) const {
  return obn2lbn(GenUnDiffBn(output_diff_bn));
}
std::string Operator::mdbn2lbn(const std::string& model_diff_bn) const {
  return mbn2lbn(GenUnDiffBn(model_diff_bn));
}
std::string Operator::ibn2lbn(const std::string& input_bn) const {
  auto it = special_ibn2lbn_.find(input_bn);
  if (it == special_ibn2lbn_.end()) {
    return normal_ibn2lbn(input_bn);
  } else {
    return it->second;
  }
}

Shape* Operator::GetShapePtr(const std::string& bn_in_op) const {
  return bn_in_op2shape_ptr_.at(bn_in_op);
}

void Operator::SetShapePtr(const std::string& bn_in_op, Shape* ptr) const {
  bn_in_op2shape_ptr_.at(bn_in_op) = ptr;
}

void Operator::SetNull4AllShapePtr() const {
  for (auto& pair : bn_in_op2shape_ptr_) {
    pair.second = nullptr;
  }
}

void Operator::EnrollDataTmpBn(const std::string& dtbn) {
  EnrollBn(&data_tmp_bns_, dtbn);
}

void Operator::EnrollInputBn(const std::string& ibn, bool has_diff) {
  EnrollBn(&input_bns_, ibn);
  if (has_diff) {
    EnrollBn(&input_diff_bns_, GenDiffBn(ibn));
  }
}

void Operator::EnrollOutputBn(const std::string& obn, bool has_diff) {
  EnrollBn(&output_bns_, obn);
  if (has_diff) {
    EnrollBn(&output_diff_bns_, GenDiffBn(obn));
  }
}

void Operator::EnrollModelBn(const std::string& mbn) {
  EnrollBn(&model_bns_, mbn);
  EnrollBn(&model_diff_bns_, GenDiffBn(mbn));
}

void Operator::EnrollModelTmpBn(const std::string& mtbn) {
  EnrollBn(&model_tmp_bns_, mtbn);
}

void Operator::EnrollBn(std::vector<std::string>* bn_vec,
                        const std::string& bn) {
  bn_vec->push_back(bn);
  CHECK(bn_in_op2shape_ptr_.emplace(bn, nullptr).second);
}

std::string UserOperator::normal_ibn2lbn(const std::string& input_bn) const {
  return GetValueFromPbOpConf(input_bn);
}
std::string UserOperator::obn2lbn(const std::string& output_bn) const {
  return op_name() + "/" + GetValueFromPbOpConf(output_bn);
}
std::string UserOperator::mtbn2lbn(const std::string& model_tmp_bn) const {
  return op_name() + "/" + model_tmp_bn;
}
std::string UserOperator::mbn2lbn(const std::string& model_bn) const {
  return op_name() + "/" + model_bn;
}

} // namespace oneflow
