#include "operator/operator.h"

namespace oneflow {
  
void Operator::OperatorFromOperatorProto(const OperatorProto& operatorproto) {
  op_conf_ = operatorproto.user_conf();
  
  GPMap2HashMap(operatorproto.special_ibn2lbn(), special_ibn2lbn_);
  //repeated string data_tmp_bns = 4
  PbRepeatedPtrField2Vec(operatorproto.data_tmp_bns(), data_tmp_bns_);
  //repeated string input_bns = 5
  PbRepeatedPtrField2Vec(operatorproto.input_bns(), input_bns_);
  //repeated string input_diff_bns = 6;
  PbRepeatedPtrField2Vec(operatorproto.input_diff_bns(), input_diff_bns_);
  //repeated string output_bns = 7
  PbRepeatedPtrField2Vec(operatorproto.output_bns(), output_bns_);
  //repeated string output_diff_bns = 8
  PbRepeatedPtrField2Vec(operatorproto.output_diff_bns(), output_diff_bns_);
  //repeated string model_bns = 9
  PbRepeatedPtrField2Vec(operatorproto.model_bns(), model_bns_);
  //repeated string model_diff_bns = 10
  PbRepeatedPtrField2Vec(operatorproto.model_diff_bns(), model_diff_bns_);
  //repeated string model_tmp_bns = 11
  PbRepeatedPtrField2Vec(operatorproto.model_tmp_bns(), model_tmp_bns_);
}

OperatorProto Operator::ToOperatorProto() {
  OperatorProto operatorproto;
  *(operatorproto.mutable_user_conf()) = op_conf_;
  *(operatorproto.mutable_special_ibn2lbn()) = HashMap2GPMap(special_ibn2lbn_);
  *(operatorproto.mutable_data_tmp_bns()) = Vec2PbRepeatedPtrField(data_tmp_bns_);
  *(operatorproto.mutable_input_bns()) = Vec2PbRepeatedPtrField(input_bns_);
  *(operatorproto.mutable_input_diff_bns()) = Vec2PbRepeatedPtrField(input_diff_bns_);
  *(operatorproto.mutable_output_bns()) = Vec2PbRepeatedPtrField(output_bns_);
  *(operatorproto.mutable_output_diff_bns()) = Vec2PbRepeatedPtrField(output_diff_bns_);
  *(operatorproto.mutable_model_bns()) = Vec2PbRepeatedPtrField(model_bns_);
  *(operatorproto.mutable_model_diff_bns()) = Vec2PbRepeatedPtrField(model_diff_bns_);
  *(operatorproto.mutable_model_tmp_bns()) = Vec2PbRepeatedPtrField(model_tmp_bns_);
  return operatorproto;
}

std::string GenDiffBn(const std::string& bn) {
  return bn + "_diff";
}

std::string GenUnDiffBn(const std::string& diff_bn) {
  CHECK_STREQ(diff_bn.substr(diff_bn.size() - 5).c_str(), "_diff");
  return diff_bn.substr(0, diff_bn.size() - 5);
}

std::string Operator::dtbn2lbn(const std::string& data_tmp_bn) const {
//  return op_name_ + "/" + data_tmp_bn;
  TODO();
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
