#include "oneflow/core/operator/reduce_concat_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ReduceConcatOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_concat_conf());
  EnrollRepeatedInputBn("in", false);
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
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  const DataType data_type = first_in_blob->data_type();
  const ReduceConcatOpConf& conf = op_conf().reduce_concat_conf();
  for (int32_t i = 1; i < conf.in_size(); ++i) {
    CHECK_EQ_OR_RETURN(data_type, GetBlobDesc4BnInOp(input_bns().Get(i))->data_type());
  }

  BlobDesc* out_blob = GetBlobDesc4BnInOp(SoleObn());
  *out_blob = *first_in_blob;
  const int64_t data_type_byte_size =
      static_cast<int64_t>(GetSizeOfDataType(first_in_blob->data_type()));
  CHECK_EQ_OR_RETURN(conf.out_size() % data_type_byte_size, 0);
  const int64_t out_blob_elem_cnt =
      RoundUp(conf.out_size() / data_type_byte_size, parallel_ctx->parallel_num());
  out_blob->mut_shape() = Shape({out_blob_elem_cnt});

  // Check valid (but can be delete)
  {
    int64_t in_blob_body_size_sum = 0;
    for (int32_t i = 0; i < conf.in_size(); ++i) {
      in_blob_body_size_sum +=
          RtBlobDesc(*(GetBlobDesc4BnInOp(input_bns().Get(i)))).AlignedByteSizeOfBlobBody();
    }
    CHECK_EQ(in_blob_body_size_sum, conf.out_size());
  }

  return Maybe<void>::Ok();
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
