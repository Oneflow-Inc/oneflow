#include "oneflow/core/operator/axpy_op.h"

namespace oneflow {

void AxpyOp::InitFromOpConf() {
  EnrollInputBn("y")->set_is_mutable(true);
  EnrollInputBn("x", false);
}

void AxpyOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  CHECK(*GetBlobDesc4BnInOp("x") == *GetBlobDesc4BnInOp("y"));
}

const PbMessage& AxpyOp::GetCustomizedConf() const { return op_conf().axpy_conf(); }

REGISTER_OP(OperatorConf::kAxpyConf, AxpyOp);

}  // namespace oneflow
