#include "operator/operator.h"

namespace oneflow {

std::string GenDiffBn(const std::string& bn) {
  return bn + "_diff";
}
std::string GenUnDiffBn(const std::string& diff_bn) {
  CHECK_STREQ(diff_bn.substr(diff_bn.size() - 5).c_str(),
              "_diff");
  return diff_bn.substr(0, diff_bn.size() - 5);
}

std::string Operator::dtbn2lbn(const std::string& data_tmp_bn) const {
  return op_name_ + "/" + data_tmp_bn;
}

BlobDesc* Operator::GetBlobDescPtr(const std::string& bn_in_op) const {
  return bn_in_op2blob_desc_ptr_.at(bn_in_op);
}

void Operator::SetBlobDescPtr(const std::string& bn_in_op,
                              BlobDesc* ptr) {
  bn_in_op2blob_desc_ptr_.at(bn_in_op) = ptr;
}

void Operator::SetNull4AllBlobDescPtr() {
  for (auto& pair : bn_in_op2blob_desc_ptr_) {
    pair.second = nullptr;
  }
}

void Operator::EnrollDataTmpBn(const std::string& dtbn) {
  data_tmp_bns_.push_back(dtbn);
  bn_in_op2blob_desc_ptr_[dtbn] = nullptr;
}
void Operator::EnrollInputBn(const std::string& ibn) {
  input_bns_.push_back(ibn);
  bn_in_op2blob_desc_ptr_[ibn] = nullptr;
}
void Operator::EnrollInputDiffBn(const std::string& idbn) {
  input_diff_bns_.push_back(idbn);
  bn_in_op2blob_desc_ptr_[idbn] = nullptr;
}
void Operator::EnrollOutputBn(const std::string& obn) {
  output_bns_.push_back(obn);
  bn_in_op2blob_desc_ptr_[obn] = nullptr;
}
void Operator::EnrollOutputDiffBn(const std::string& odbn) {
  output_diff_bns_.push_back(odbn);
  bn_in_op2blob_desc_ptr_[odbn] = nullptr;
}

void Operator::EnrollModelBn(const std::string& mbn) {
  model_bns_.push_back(mbn);
  bn_in_op2blob_desc_ptr_[mbn] = nullptr;
}
void Operator::EnrollModelDiffBn(const std::string& mdbn) {
  model_diff_bns_.push_back(mdbn);
  bn_in_op2blob_desc_ptr_[mdbn] = nullptr;
}
void Operator::EnrollModelTmpBn(const std::string& mtbn) {
  model_tmp_bns_.push_back(mtbn);
  bn_in_op2blob_desc_ptr_[mtbn] = nullptr;
}

std::string UserOperator::ibn2lbn(const std::string& input_bn) const {
  return GetValueFromPbOpConf(input_bn);
}
std::string UserOperator::obn2lbn(const std::string& output_bn) const {
  return op_name() + "/" + GetValueFromPbOpConf(output_bn);
}
std::string UserOperator::idbn2lbn(const std::string& input_diff_bn) const {
  return ibn2lbn(GenUnDiffBn(input_diff_bn));
}
std::string UserOperator::odbn2lbn(const std::string& output_diff_bn) const {
  return obn2lbn(GenUnDiffBn(output_diff_bn));
}
std::string UserOperator::mtbn2lbn(const std::string& model_tmp_bn) const {
  return op_name() + "/" + model_tmp_bn;
}
std::string UserOperator::mbn2lbn(const std::string& model_bn) const {
  return op_name() + "/" + model_bn;
}
std::string UserOperator::mdbn2lbn(const std::string& model_diff_bn) const {
  return mbn2lbn(GenUnDiffBn(model_diff_bn));
}

} // namespace oneflow
