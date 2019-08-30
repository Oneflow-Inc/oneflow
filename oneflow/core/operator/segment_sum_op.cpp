#include "oneflow/core/operator/segment_sum_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SegmentSumOp::InitFromOpConf() {
  CHECK(op_conf().has_segment_sum_conf());
  EnrollInputBn("in");
  EnrollInputBn("segment_ids", false);
  EnrollInputBn("unique_segment_ids", false);
  EnrollOutputBn("out");
}

const PbMessage& SegmentSumOp::GetCustomizedConf() const { return op_conf().segment_sum_conf(); }

void SegmentSumOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  const BlobDesc* segment_ids_blob = GetBlobDesc4BnInOp("segment_ids");
  const BlobDesc* unique_segment_ids_blob = GetBlobDesc4BnInOp("unique_segment_ids");

  const int32_t unique_ids_count = unique_segment_ids_blob->shape().At(0);
  // input data's dim0 == segment ids' dim0
  CHECK_EQ(in_blob->shape().At(0), segment_ids_blob->shape().At(0));
  // segmnet_ids must be 1D tensor
  CHECK_EQ(segment_ids_blob->shape().NumAxes(), 2);
  auto dims = in_blob->shape().dim_vec();
  dims[0] = unique_ids_count;
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  out_blob->set_data_type(in_blob->data_type());
  out_blob->mut_shape() = Shape(dims);
}

void SegmentSumOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn("in").shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kSegmentSumConf, SegmentSumOp);

}  // namespace oneflow
