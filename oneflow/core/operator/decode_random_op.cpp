#include "oneflow/core/operator/decode_random_op.h"

namespace oneflow {

void DecodeRandomOp::InitFromOpConf() {
  CHECK(op_conf().has_decode_random_conf());
  if (op_conf().decode_random_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

const PbMessage& DecodeRandomOp::GetCustomizedConf() const {
  return op_conf().decode_random_conf();
}

void DecodeRandomOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->mutable_decode_random_conf()->set_random_seed(NewRandomSeed());
}

Maybe<void> DecodeRandomOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const DecodeRandomOpConf& conf = op_conf().decode_random_conf();
  std::vector<int64_t> dim_vec(1 + conf.shape().dim_size());
  int64_t batch_size = conf.batch_size();
  OF_CHECK_GE(batch_size, parallel_ctx->parallel_num());
  CHECK_EQ_OR_RETURN(batch_size % parallel_ctx->parallel_num(), 0);
  dim_vec[0] = batch_size / parallel_ctx->parallel_num();
  FOR_RANGE(size_t, j, 1, dim_vec.size()) { dim_vec[j] = conf.shape().dim(j - 1); }
  out_blob_desc->mut_shape() = Shape(dim_vec);
  out_blob_desc->set_data_type(conf.data_type());
  out_blob_desc->set_has_data_id_field(job_desc().SizeOfOneDataId() > 0);
  return Maybe<void>::Ok();
}

Maybe<void> DecodeRandomOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("out")->set_value(0);
  return Maybe<void>::Ok();
}

Maybe<void> DecodeRandomOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kDecodeRandomConf, DecodeRandomOp);

}  // namespace oneflow
