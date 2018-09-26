#include "oneflow/core/operator/batch_permutation_op.h"

namespace oneflow {

void BatchPermutationOp::InitFromOpConf() {
  CHECK(op_conf().has_batch_permutation_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& BatchPermutationOp::GetCustomizedConf() const {
  return op_conf().batch_permutation_conf();
}

void BatchPermutationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  if (op_conf().batch_permutation_conf().data_format() != "channels_first"
      || in_blob_desc->shape().NumAxes() != 4) {
    LOG(FATAL) << "batch_permutation only supports NCHW";
  }
  const int32_t scale = op_conf().batch_permutation_conf().scale();
  CHECK_GT(scale, 1);
  out_blob_desc->mut_shape() =
      Shape({in_blob_desc->shape().At(0), in_blob_desc->shape().At(1),
             scale * in_blob_desc->shape().At(2), scale * in_blob_desc->shape().At(3)});
}

REGISTER_OP(OperatorConf::kBatchPermutationConf, BatchPermutationOp);

}  // namespace oneflow
