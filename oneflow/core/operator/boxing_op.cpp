#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void BoxingOp::InitFromOpConf() {
  CHECK(op_conf().has_boxing_conf());
  TODO();
}

const PbMessage& BoxingOp::GetSpecialConf() const {
  return op_conf().boxing_conf();
}

std::string BoxingOp::ibn2lbn(const std::string& input_bn) const {
  return GetStringFromSpecialConf("lbn");
}

std::string BoxingOp::obn2lbn(const std::string& output_bn) const {
  return GetStringFromSpecialConf("lbn");
}

void BoxingOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) {
  TODO();
}

REGISTER_OP(OperatorConf::kBoxingConf, BoxingOp);

}  // namespace oneflow
