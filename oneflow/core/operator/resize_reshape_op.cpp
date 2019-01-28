#include "oneflow/core/operator/resize_reshape_op.h"

namespace oneflow {

void ResizeReshapeOp::InitFromOpConf() {
  CHECK(op_conf().has_resize_reshape_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ResizeReshapeOp::GetCustomizedConf() const {
  return op_conf().resize_reshape_conf();
}

void ResizeReshapeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->mut_shape() = Shape(op_conf().resize_reshape_conf().shape());
}

bool ResizeReshapeOp::IsInputBlobAllowedModelSplit(const std::string& ibn) const {
  CHECK(std::find(input_bns().begin(), input_bns().end(), ibn) != input_bns().end());
  return false;
}

void ResizeReshapeOp::InferOutputBlobModelSplitAxis(
    std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
    std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
    const ParallelContext* parallel_context) const {
  UNIMPLEMENTED();
}

REGISTER_OP(OperatorConf::kResizeReshapeConf, ResizeReshapeOp);

}  // namespace oneflow
