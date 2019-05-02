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
  if (Global<JobDesc>::Get()->IsPredict()
      && Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    return new NormalForwardLogicalNode;
  } else {
    return new ReduceConcatLogicalNode;
  }
}

void ReduceConcatOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx, int64_t record_piece_size,
                                    std::function<void(OpContext*)> EnrollOpCtx) const {
  const BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  const DataType data_type = first_in_blob->data_type();
  for (int32_t i = 1; i < op_conf().reduce_concat_conf().in_num(); ++i) {
    CHECK_EQ(data_type, GetBlobDesc4BnInOp(input_bns().Get(i))->data_type());
  }

  BlobDesc* out_blob = GetBlobDesc4BnInOp(SoleObn());
  *out_blob = *first_in_blob;
  int64_t in_blob_body_size_sum = 0;
  for (int32_t i = 0; i < op_conf().reduce_concat_conf().in_num(); ++i) {
    in_blob_body_size_sum +=
        RtBlobDesc(*(GetBlobDesc4BnInOp(input_bns().Get(i)))).ByteSizeOfBlobBody();
  }
  const int64_t data_type_byte_size =
      static_cast<int64_t>(GetSizeOfDataType(first_in_blob->data_type()));
  CHECK_EQ(in_blob_body_size_sum % data_type_byte_size, 0);
  const int64_t out_blob_elem_cnt =
      RoundUp(in_blob_body_size_sum / data_type_byte_size, parallel_ctx->parallel_num());
  out_blob->mut_shape() = Shape({out_blob_elem_cnt});

  // construct reduce_concat_op_ctx for later CHECK in ReduceConcatOp::VirtualGenKernelConf
  ReduceConcatOpCtx* reduce_concat_op_ctx = new ReduceConcatOpCtx(out_blob_elem_cnt);
  EnrollOpCtx(reduce_concat_op_ctx);
}

void ReduceConcatOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  ReduceConcatKernelConf* reduce_concat_conf = kernel_conf->mutable_reduce_concat_conf();
  int64_t offset = 0;
  for (int32_t i = 0; i < op_conf().reduce_concat_conf().in_num(); ++i) {
    reduce_concat_conf->mutable_data_offset()->Add(offset);
    offset += RtBlobDesc(*(GetBlobDesc4BnInOp(input_bns().Get(i)))).ByteSizeOfBlobBody();
  }
  const int64_t data_type_byte_size =
      static_cast<int64_t>(GetSizeOfDataType(GetBlobDesc4BnInOp(input_bns().Get(0))->data_type()));
  CHECK_EQ(offset % data_type_byte_size, 0);
  const int64_t out_blob_elem_cnt =
      RoundUp(offset / data_type_byte_size, parallel_ctx->parallel_num());
  const ReduceConcatOpCtx* reduce_concat_op_ctx = static_cast<const ReduceConcatOpCtx*>(op_ctx);
  CHECK_EQ(reduce_concat_op_ctx->out_blob_elem_cnt, out_blob_elem_cnt);
}

LogicalBlobId ReduceConcatOp::ibn2lbi(const std::string& input_bn) const {
  if (Global<JobDesc>::Get()->IsPredict()
      && Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    return this->Operator::ibn2lbi(input_bn);
  } else {
    return GenPackedLbi();
  }
}

LogicalBlobId ReduceConcatOp::obn2lbi(const std::string& output_bn) const {
  if (Global<JobDesc>::Get()->IsPredict()
      && Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    return this->Operator::obn2lbi(output_bn);
  } else {
    LogicalBlobId ret;
    ret.set_op_name(op_name());
    ret.set_blob_name("out");
    return ret;
  }
}

void ReduceConcatOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  for (const auto& ibn : input_bns()) { CHECK_EQ(*HasBatchDim4BnInOp(ibn), false); }
  *HasBatchDim4BnInOp("out") = false;
}

void ReduceConcatOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  for (const auto& ibn : input_bns()) {
    CHECK(SbpInferHint4Ibn(ibn).sbp_parallel().has_partial_sum_parallel());
  }
  SbpSignatureBuilder().PartialSum(input_bns()).PartialSum(output_bns()).Build(sbp_signature);
}

REGISTER_OP(OperatorConf::kReduceConcatConf, ReduceConcatOp);

}  // namespace oneflow
