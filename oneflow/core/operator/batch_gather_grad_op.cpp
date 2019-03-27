#include "oneflow/core/operator/batch_gather_grad_op.h"

namespace oneflow {

void BatchGatherGradOp::InitFromOpConf() {
  CHECK(op_conf().has_batch_gather_grad_conf());
  EnrollInputBn("indices", false);
  EnrollInputBn("out_diff", false);
  EnrollOutputBn("in_diff", false);
}

const PbMessage& BatchGatherGradOp::GetCustomizedConf() const {
  return op_conf().batch_gather_grad_conf();
}

void BatchGatherGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const int64_t gather_dim_size = op_conf().batch_gather_grad_conf().gather_dim_size();
  CHECK_GE(gather_dim_size, 1);
  const BlobDesc* out_diff = GetBlobDesc4BnInOp("out_diff");
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK(IsIntegralDataType(indices->data_type()));
  CHECK_GE(indices->shape().NumAxes(), 1);
  CHECK_GE(out_diff->shape().NumAxes(), indices->shape().NumAxes());
  std::vector<int64_t> in_diff_dim_vec;
  in_diff_dim_vec.insert(in_diff_dim_vec.end(), indices->shape().dim_vec().cbegin(),
                         indices->shape().dim_vec().cend() - 1);
  in_diff_dim_vec.push_back(gather_dim_size);
  in_diff_dim_vec.insert(in_diff_dim_vec.end(),
                         out_diff->shape().dim_vec().cbegin() + indices->shape().NumAxes(),
                         out_diff->shape().dim_vec().cend());
  BlobDesc* in_diff = GetBlobDesc4BnInOp("in_diff");
  *in_diff = *out_diff;
  in_diff->mut_shape() = Shape(in_diff_dim_vec);
}

void BatchGatherGradOp::GetSbpSignatures(
    std::vector<std::unique_ptr<const SbpSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(MakeDataSplitSbpSignature(this));
}

REGISTER_OP(OperatorConf::kBatchGatherGradConf, BatchGatherGradOp);

}  // namespace oneflow
