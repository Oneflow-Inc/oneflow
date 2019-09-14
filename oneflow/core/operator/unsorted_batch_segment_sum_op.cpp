#include "oneflow/core/operator/unsorted_batch_segment_sum_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void UnsortedBatchSegmentSumOp::InitFromOpConf() {
  CHECK(op_conf().has_unsorted_batch_segment_sum_conf());
  EnrollInputBn("segment_ids", false);
  EnrollInputBn("data");
  EnrollOutputBn("out");
}

const PbMessage& UnsortedBatchSegmentSumOp::GetCustomizedConf() const {
  return op_conf().unsorted_batch_segment_sum_conf();
}

Maybe<void> UnsortedBatchSegmentSumOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const int64_t num_segments = op_conf().unsorted_batch_segment_sum_conf().num_segments();
  CHECK_GE_OR_RETURN(num_segments, 1);
  const BlobDesc* data = GetBlobDesc4BnInOp("data");
  const BlobDesc* segment_ids = GetBlobDesc4BnInOp("segment_ids");
  CHECK_OR_RETURN(IsIntegralDataType(segment_ids->data_type()));
  CHECK_GE_OR_RETURN(segment_ids->shape().NumAxes(), 1);
  CHECK_GE_OR_RETURN(data->shape().NumAxes(), segment_ids->shape().NumAxes());
  std::vector<int64_t> out_dim_vec;
  out_dim_vec.insert(out_dim_vec.end(), segment_ids->shape().dim_vec().cbegin(),
                     segment_ids->shape().dim_vec().cend() - 1);
  out_dim_vec.push_back(num_segments);
  out_dim_vec.insert(out_dim_vec.end(),
                     data->shape().dim_vec().cbegin() + segment_ids->shape().NumAxes(),
                     data->shape().dim_vec().cend());
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *data;
  out->mut_shape() = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> UnsortedBatchSegmentSumOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t indices_num_axes = JUST(LogicalBlobDesc4Ibn("segment_ids"))->shape().NumAxes();
  OF_CHECK_GT(indices_num_axes, 1) << "UnsortedBatchSegmentSumOp: indices_num_axes equals "
                                   << indices_num_axes << " (should be bigger than 1).";
  FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
    SbpSignatureBuilder()
        .Split("segment_ids", i)
        .Split("data", i)
        .Split("out", i)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kUnsortedBatchSegmentSumConf, UnsortedBatchSegmentSumOp);

}  // namespace oneflow
