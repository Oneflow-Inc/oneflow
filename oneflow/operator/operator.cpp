#include "operator/operator.h"

namespace oneflow {

namespace {

inline std::string GenDiffBn(const std::string& bn) {
  return bn + "_diff";
}

inline std::string GenUnDiffBn(const std::string& diff_bn) {
  CHECK_STREQ(diff_bn.substr(diff_bn.size() - 5).c_str(), "_diff");
  return diff_bn.substr(0, diff_bn.size() - 5);
}

} // namespace

std::string Operator::dtbn2lbn(const std::string& data_tmp_bn) const {
  return op_name_ + "/" + data_tmp_bn;
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

std::string Operator::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(*pb_op_conf_, k);
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
