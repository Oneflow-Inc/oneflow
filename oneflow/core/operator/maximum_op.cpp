#include "oneflow/core/operator/maximum_op.h"

namespace oneflow {

void MaximumOp::VirtualInitFromOpConf() {
  CHECK(op_conf().has_maximum_conf());
  EnrollTmpBn("mask");
}

const PbMessage& MaximumOp::GetCustomizedConf() const { return op_conf().maximum_conf(); }

Maybe<void> MaximumOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(0));
  BlobDesc* mask_blob_desc = GetBlobDesc4BnInOp("mask");
  *mask_blob_desc = *in_0_blob_desc;
  mask_blob_desc->set_data_type(DataType::kInt32);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kMaximumConf, MaximumOp);

}  // namespace oneflow
