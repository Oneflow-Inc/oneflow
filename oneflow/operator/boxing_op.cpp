#include "operator/boxing_op.h"

namespace oneflow {

void BoxingOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_boxing_op_conf());
  auto cnf = new BoxingOpConf(op_conf.boxing_op_conf());
  mut_pb_op_conf().reset(cnf);

  for (int32_t i = 0; i < cnf->in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i));
  }
  EnrollDataTmpBn("middle");
  for (int32_t i = 0; i < cnf->out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
  }
}

std::string BoxingOp::normal_ibn2lbn(const std::string& input_bn) const {
  return GetValueFromPbOpConf("lbn");
}

std::string BoxingOp::obn2lbn(const std::string& output_bn) const {
  return GetValueFromPbOpConf("lbn");
}

} // namespace oneflow
