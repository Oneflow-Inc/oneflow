#include "oneflow/core/operator/gelu_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void GeluOp::InitFromOpConf() {
  CHECK(op_conf().has_gelu_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& GeluOp::GetCustomizedConf() const { return op_conf().gelu_conf(); }

void GeluOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kGeluConf, GeluOp);

}  // namespace oneflow
