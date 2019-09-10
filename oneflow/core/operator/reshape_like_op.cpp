#include "oneflow/core/operator/reshape_like_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/reshape_util.h"

namespace oneflow {

void ReshapeLikeOp::InitFromOpConf() {
  CHECK(op_conf().has_reshape_like_conf());
  EnrollInputBn("x");
  EnrollOutputBn("y");
  EnrollInputBn("like", false)->set_use_header_only(true);
}

const PbMessage& ReshapeLikeOp::GetCustomizedConf() const { return op_conf().reshape_like_conf(); }

Maybe<void> ReshapeLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("x")->shape().elem_cnt(),
                     GetBlobDesc4BnInOp("like")->shape().elem_cnt());
  GetBlobDesc4BnInOp("y")->CopyMetaFrom(*GetBlobDesc4BnInOp("like"));
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeLikeOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const auto& in_shape = JUST(LogicalBlobDesc4Ibn("x"))->shape();
  const auto& out_shape = JUST(LogicalBlobDesc4Ibn("like"))->shape();
  if (in_shape.At(0) == in_shape.At(0)) {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  HashMap<int, int> squeezed_group_start_in_axis2out_axis;
  HashMap<int, int> x_squeezed_axis2original_axis;
  HashMap<int, int> y_squeezed_axis2original_axis;
  {
    Shape squeezed_in_shape;
    Shape squeezed_out_shape;
    Squeeze(in_shape, &squeezed_in_shape, &x_squeezed_axis2original_axis);
    Squeeze(out_shape, &squeezed_out_shape, &y_squeezed_axis2original_axis);
    GetGroupStartInAxis2OutAxis(squeezed_in_shape, squeezed_out_shape,
                                &squeezed_group_start_in_axis2out_axis);
  }
  for (const auto& pair : squeezed_group_start_in_axis2out_axis) {
    int64_t start_in_axis = x_squeezed_axis2original_axis.at(pair.first);
    int64_t start_out_axis = y_squeezed_axis2original_axis.at(pair.second);
    SbpSignatureBuilder()
        .Split(input_bns(), start_in_axis)
        .Split(output_bns(), start_out_axis)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  SbpSignatureBuilder()
      .PartialSum(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kReshapeLikeConf, ReshapeLikeOp);

}  // namespace oneflow
