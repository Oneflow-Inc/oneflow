#include "oneflow/core/operator/reshape_like_op.h"

namespace oneflow {

void ReshapeLikeOp::InitFromOpConf() {
  CHECK(op_conf().has_reshape_like_conf());
  EnrollInputBn("x");
  EnrollOutputBn("y");
  EnrollInputBn("like")->set_use_header_only(true);
}

const PbMessage& ReshapeLikeOp::GetCustomizedConf() const { return op_conf().reshape_like_conf(); }

void ReshapeLikeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  CHECK_EQ(GetBlobDesc4BnInOp("x")->shape().elem_cnt(),
           GetBlobDesc4BnInOp("like")->shape().elem_cnt());
  GetBlobDesc4BnInOp("y")->CopyMetaFrom(*GetBlobDesc4BnInOp("like"));
}

REGISTER_OP(OperatorConf::kReshapeLikeConf, ReshapeLikeOp);

}  // namespace oneflow
