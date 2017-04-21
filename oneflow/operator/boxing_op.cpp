#include "operator/boxing_op.h"

namespace oneflow {

void BoxingOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_boxing_op_conf());
  mut_op_conf() = op_conf;

  for (int64_t i = 0; i < op_conf.boxing_op_conf().in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i));
  }
  EnrollDataTmpBn("middle");
  for (int64_t i = 0; i < op_conf.boxing_op_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
  }
}

std::string BoxingOp::normal_ibn2lbn(const std::string& input_bn) const {
  return GetValueFromPbOpConf("lbn");
}

std::string BoxingOp::obn2lbn(const std::string& output_bn) const {
  return GetValueFromPbOpConf("lbn");
}
std::string BoxingOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().boxing_op_conf(), k);
}
} // namespace oneflow
