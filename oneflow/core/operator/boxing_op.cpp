#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void BoxingOp::InitFromOpConf() {
  CHECK(op_conf().has_boxing_conf());
  auto boxing_conf = op_conf().boxing_conf();

  for (int64_t i = 0; i != boxing_conf.in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i));
  }
  if (boxing_conf.has_concat_box() || boxing_conf.has_clone_box()) {
    EnrollDataTmpBn("middle");
  }
  for (int64_t i = 0; i != boxing_conf.out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
  }
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
