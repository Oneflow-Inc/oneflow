#include "oneflow/core/operator/softmax_op.h"

namespace oneflow {

void SoftmaxOp::InitFromOpConf() {
  CHECK(op_conf().has_softmax_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("tmp");
}

const PbMessage& SoftmaxOp::GetSpecialConf() const {
  return op_conf().softmax_conf();
}

void SoftmaxOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() =
      Shape({in_blob_desc->shape().At(0), in_blob_desc->shape().Count(1)});
  // tmp
  BlobDesc* tmp_blob_desc = GetBlobDesc4BnInOp("tmp");
  tmp_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0)});
  tmp_blob_desc->set_data_type(in_blob_desc->data_type());
}

REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);

}  // namespace oneflow
