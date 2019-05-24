#include "oneflow/core/operator/decode_random_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void DecodeRandomOp::InitFromOpConf() {
  CHECK(op_conf().has_decode_random_conf());
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

void DecodeRandomOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx,
                                    int64_t record_piece_size) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const DecodeRandomOpConf& conf = op_conf().decode_random_conf();
  out_blob_desc->mut_shape() = Shape(op_conf().decode_random_conf().shape());
  const BalancedSplitter bs(out_blob_desc->shape().At(conf.split_axis()),
                            parallel_ctx->parallel_num());
  out_blob_desc->set_data_type(conf.data_type());
  out_blob_desc->mut_shape().Set(conf.split_axis(), bs.At(parallel_ctx->parallel_id()).size());
}

void DecodeRandomOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = true;
}

void DecodeRandomOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  const DecodeRandomOpConf& conf = op_conf().decode_random_conf();
  (*sbp_signature->mutable_bn_in_op2sbp_parallel())["out"].mutable_split_parallel()->set_axis(
      conf.split_axis());
}

REGISTER_OP(OperatorConf::kDecodeRandomConf, DecodeRandomOp);

}  // namespace oneflow
