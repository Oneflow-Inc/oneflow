#include "oneflow/core/operator/reduce_concat_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

struct ReduceConcatOpCtx : public OpContext {
  ReduceConcatOpCtx(const int64_t elem_cnt) : out_blob_elem_cnt(elem_cnt) {}
  int64_t out_blob_elem_cnt;
};

void ReduceConcatOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_concat_conf());
  for (int32_t i = 0; i < op_conf().reduce_concat_conf().in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  EnrollOutputBn("out", false);
}

const PbMessage& ReduceConcatOp::GetCustomizedConf() const {
  return op_conf().reduce_concat_conf();
}

LogicalNode* ReduceConcatOp::NewProperLogicalNode() const {
  // TODO(): return new ReduceConcatLogicalNode;
  return new NormalForwardLogicalNode;
}

Maybe<void> ReduceConcatOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  const DataType data_type = first_in_blob->data_type();
  for (int32_t i = 1; i < op_conf().reduce_concat_conf().in_num(); ++i) {
    CHECK_EQ_OR_RETURN(data_type, GetBlobDesc4BnInOp(input_bns().Get(i))->data_type());
  }

  BlobDesc* out_blob = GetBlobDesc4BnInOp(SoleObn());
  *out_blob = *first_in_blob;
  int64_t in_blob_body_size_sum = 0;
  for (int32_t i = 0; i < op_conf().reduce_concat_conf().in_num(); ++i) {
    in_blob_body_size_sum +=
        RtBlobDesc(*(GetBlobDesc4BnInOp(input_bns().Get(i)))).AlignedByteSizeOfBlobBody();
  }
  const int64_t data_type_byte_size =
      static_cast<int64_t>(GetSizeOfDataType(first_in_blob->data_type()));
  CHECK_EQ_OR_RETURN(in_blob_body_size_sum % data_type_byte_size, 0);
  const int64_t out_blob_elem_cnt =
      RoundUp(in_blob_body_size_sum / data_type_byte_size, parallel_ctx->parallel_num());
  out_blob->mut_shape() = Shape({out_blob_elem_cnt});

  // construct reduce_concat_op_ctx for later CHECK in ReduceConcatOp::VirtualGenKernelConf
  ReduceConcatOpCtx* reduce_concat_op_ctx = new ReduceConcatOpCtx(out_blob_elem_cnt);
  EnrollOpCtx(reduce_concat_op_ctx);
  return Maybe<void>::Ok();
}

void ReduceConcatOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  ReduceConcatKernelConf* reduce_concat_conf = kernel_conf->mutable_reduce_concat_conf();
  int64_t offset = 0;
  for (int32_t i = 0; i < op_conf().reduce_concat_conf().in_num(); ++i) {
    reduce_concat_conf->mutable_data_offset()->Add(offset);
    offset += RtBlobDesc(*(GetBlobDesc4BnInOp(input_bns().Get(i)))).AlignedByteSizeOfBlobBody();
  }
  const int64_t data_type_byte_size =
      static_cast<int64_t>(GetSizeOfDataType(GetBlobDesc4BnInOp(input_bns().Get(0))->data_type()));
  CHECK_EQ(offset % data_type_byte_size, 0);
  const int64_t out_blob_elem_cnt =
      RoundUp(offset / data_type_byte_size, parallel_ctx->parallel_num());
  const ReduceConcatOpCtx* reduce_concat_op_ctx = static_cast<const ReduceConcatOpCtx*>(op_ctx);
  CHECK_EQ(reduce_concat_op_ctx->out_blob_elem_cnt, out_blob_elem_cnt);
}

Maybe<void> ReduceConcatOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& ibn : input_bns()) {
    CHECK_EQ_OR_RETURN(BatchAxis4BnInOp(ibn)->has_value(), false);
  }
  BatchAxis4BnInOp("out")->clear_value();
  return Maybe<void>::Ok();
}

Maybe<void> ReduceConcatOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  for (const auto& ibn : input_bns()) {
    CHECK_OR_RETURN(JUST(SbpInferHint4Ibn(ibn))->sbp_parallel().has_partial_sum_parallel());
  }
  SbpSignatureBuilder().PartialSum(input_bns()).PartialSum(output_bns()).Build(sbp_signature);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kReduceConcatConf, ReduceConcatOp);

}  // namespace oneflow
