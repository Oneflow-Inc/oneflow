#include "oneflow/core/operator/gather_grad_op.h"
#include "oneflow/core/operator/gather_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void GatherGradOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_grad_conf());
  EnrollInputBn("indices", false);
  EnrollInputBn("out_diff", false);
  EnrollOutputBn("in_diff", false);
}

const PbMessage& GatherGradOp::GetCustomizedConf() const { return op_conf().gather_grad_conf(); }

Maybe<void> GatherGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const GatherGradOpConf& conf = op_conf().gather_grad_conf();
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK_OR_RETURN(IsIntegralDataType(indices->data_type()));
  const BlobDesc* out_diff = GetBlobDesc4BnInOp("out_diff");
  std::vector<int64_t> in_diff_dim_vec;
  in_diff_dim_vec.insert(in_diff_dim_vec.end(), out_diff->shape().dim_vec().cbegin(),
                         out_diff->shape().dim_vec().cbegin() + conf.axis());
  in_diff_dim_vec.push_back(conf.gather_dim_size());
  in_diff_dim_vec.insert(
      in_diff_dim_vec.end(),
      out_diff->shape().dim_vec().cbegin() + conf.axis() + indices->shape().NumAxes(),
      out_diff->shape().dim_vec().end());
  BlobDesc* in_diff = GetBlobDesc4BnInOp("in_diff");
  in_diff->set_data_type(out_diff->data_type());
  in_diff->mut_shape() = Shape(in_diff_dim_vec);
  return Maybe<void>::Ok();
}

void GatherGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t gather_axis = op_conf().gather_grad_conf().axis();
  const int64_t indices_num_axes = LogicalBlobDesc4Ibn("indices").shape().NumAxes();
  const int64_t out_diff_num_axes = LogicalBlobDesc4Ibn("out_diff").shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, indices_num_axes) {
    SbpSignatureBuilder()
        .Split("indices", i)
        .Split("out_diff", i + gather_axis)
        .PartialSum("in_diff")
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  FOR_RANGE(int64_t, i, 0, out_diff_num_axes) {
    if (i >= gather_axis && i < gather_axis + indices_num_axes) { continue; }
    const int64_t in_diff_split_axis = (i < gather_axis) ? i : i - indices_num_axes + 1;
    if (in_diff_split_axis == gather_axis) { continue; }
    SbpSignatureBuilder()
        .Broadcast("indices")
        .Split("out_diff", i)
        .Split("in_diff", in_diff_split_axis)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  SbpSignatureBuilder()
      .Broadcast("indices")
      .PartialSum("out_diff")
      .PartialSum("in_diff")
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

Maybe<void> GatherGradOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("in_diff") = false;
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kGatherGradConf, GatherGradOp);

}  // namespace oneflow
