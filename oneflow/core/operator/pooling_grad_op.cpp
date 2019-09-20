#include "oneflow/core/operator/pooling_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void PoolingGradOp::InitFromOpConf() {
  const PoolingConf& pooling_conf = this->op_conf().pooling_grad_conf().pooling_conf();
  std::string pool_mode = pooling_conf.pool_mode();
  std::transform(pool_mode.begin(), pool_mode.end(), pool_mode.begin(), ::tolower);
  if (pool_mode != "avg" && pool_mode != "max") {
    LOG(FATAL) << "Invalid pooling type in " << op_name() << ". It must be avg or max";
  }
  CheckPoolSizeAndStrides();
  EnrollInputBn("x");
  EnrollInputBn("y");
  EnrollInputBn("dy");
  EnrollOutputBn("dx");
}

const PbMessage& PoolingGradOp::GetCustomizedConf() const { return op_conf().pooling_grad_conf(); }

Maybe<void> PoolingGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
  CHECK_GE_OR_RETURN(x_blob_desc->shape().NumAxes(), 3);
  CHECK_LE_OR_RETURN(x_blob_desc->shape().NumAxes(), 5);
  // out
  *GetBlobDesc4BnInOp("dx") = *x_blob_desc;
  return Maybe<void>::Ok();
}

void PoolingGradOp::CheckPoolSizeAndStrides() const {
  const PoolingConf& pooling_conf = this->op_conf().pooling_grad_conf().pooling_conf();
  const int32_t num_spatial_dims = pooling_conf.num_spatial_dims();
  const PbRf<int32_t>& pool_size = pooling_conf.pool_size();
  CHECK_EQ(pool_size.size(), num_spatial_dims);
  for (int32_t pool_dim : pool_size) { CHECK_GT(pool_dim, 0); }
  const PbRf<int32_t>& strides = pooling_conf.strides();
  CHECK_EQ(strides.size(), num_spatial_dims);
  for (int32_t stride_dim : strides) { CHECK_GT(stride_dim, 0); }
}

Maybe<void> PoolingGradOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kPoolingGradConf, PoolingGradOp);

}  // namespace oneflow
