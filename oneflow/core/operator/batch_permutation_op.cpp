#include "oneflow/core/operator/batch_permutation_op.h"

namespace oneflow {

void BatchPermutationOp::InitFromOpConf() {
  CHECK(op_conf().has_batch_permutation_conf());
  EnrollInputBn("in");
  EnrollInputBn("indices", false);
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
  const BlobDesc* indices_blob_desc = GetBlobDesc4BnInOp("indices");
  if (op_conf().batch_permutation_conf().data_format() != "channels_first"
      || in_blob_desc->shape().NumAxes() != 4) {
    LOG(FATAL) << "batch_permutation only supports NCHW";
  }
  if (indices_blob_desc->shape().NumAxes() != 1) {
    FOR_RANGE(int32_t, idx, 1, indices_blob_desc->shape().NumAxes()) {
      CHECK_EQ(indices_blob_desc->shape().At(idx), 1);
    }
  }
  CHECK_EQ(indices_blob_desc->data_type(), DataType::kInt32);
  CHECK_EQ(indices_blob_desc->shape().At(0), in_blob_desc->shape().At(0));
  if (indices_blob_desc->has_varying_instance_num_field()
      || in_blob_desc->has_varying_instance_num_field()) {
    CHECK_EQ(indices_blob_desc->instance_inner_shape().At(0), 1);
    CHECK_EQ(in_blob_desc->instance_inner_shape().At(0), 1);
  }
}

REGISTER_OP(OperatorConf::kBatchPermutationConf, BatchPermutationOp);

}  // namespace oneflow
