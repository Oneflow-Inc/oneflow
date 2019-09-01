#include "oneflow/core/operator/record_load_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void RecordLoadOp::InitFromOpConf() {
  CHECK(op_conf().has_record_load_conf());
  if (op_conf().record_load_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

const PbMessage& RecordLoadOp::GetCustomizedConf() const { return op_conf().record_load_conf(); }

Maybe<void> RecordLoadOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, int64_t record_piece_size) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  CHECK_EQ_OR_RETURN(record_piece_size % parallel_ctx->parallel_num(), 0);
  out_blob_desc->mut_shape() = Shape({record_piece_size / parallel_ctx->parallel_num()});
  out_blob_desc->set_data_type(kOFRecord);
  return Maybe<void>::Ok();
}

void RecordLoadOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  int64_t device_piece_size = GetBlobDesc4BnInOp("out")->shape().At(0);
  kernel_conf->mutable_record_load_conf()->set_device_piece_size(device_piece_size);
}

Maybe<void> RecordLoadOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = true;
  return Maybe<void>::Ok();
}

void RecordLoadOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_CPU_OP(OperatorConf::kRecordLoadConf, RecordLoadOp);

}  // namespace oneflow
