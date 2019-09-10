#include "oneflow/core/operator/reshape_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/reshape_util.h"

namespace oneflow {

void ReshapeOp::InitFromOpConf() {
  CHECK(op_conf().has_reshape_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_const_inplace_ibn("in");
}

const PbMessage& ReshapeOp::GetCustomizedConf() const { return op_conf().reshape_conf(); }

Maybe<void> ReshapeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *out_blob_desc = *in_blob_desc;
  const ReshapeOpConf& conf = op_conf().reshape_conf();
  CHECK_GE_OR_RETURN(conf.shape().dim_size(), 1);
  std::vector<int64_t> dim_vec = {conf.shape().dim().begin(), conf.shape().dim().end()};
  for (int32_t i = 0; i < dim_vec.size(); ++i) { CHECK_GT_OR_RETURN(dim_vec[i], 0); }
  const auto& sbp_parallel_it = sbp_signature->bn_in_op2sbp_parallel().find("out");
  CHECK_OR_RETURN(sbp_parallel_it != sbp_signature->bn_in_op2sbp_parallel().end());
  const SbpParallel& sbp_parallel = sbp_parallel_it->second;
  if (sbp_parallel.has_split_parallel()) {
    const int64_t split_axis = sbp_parallel.split_parallel().axis();
    BalancedSplitter splitter(conf.shape().dim().Get(split_axis), parallel_ctx->parallel_num());
    CHECK_GE_OR_RETURN(conf.shape().dim().Get(split_axis), parallel_ctx->parallel_num());
    dim_vec[split_axis] = splitter.At(parallel_ctx->parallel_id()).size();
  }
  out_blob_desc->mut_shape() = Shape(dim_vec);
  CHECK_EQ_OR_RETURN(out_blob_desc->shape().elem_cnt(), in_blob_desc->shape().elem_cnt());
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const auto& in_shape = JUST(LogicalBlobDesc4Ibn("in"))->shape();
  const auto& out_shape = JUST(GetLogicalOutBlobShape(in_shape, op_conf().reshape_conf().shape()));
  if (in_shape.At(0) == in_shape.At(0)) {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  HashMap<int, int> squeezed_group_start_in_axis2out_axis;
  HashMap<int, int> in_squeezed_axis2original_axis;
  HashMap<int, int> out_squeezed_axis2original_axis;
  {
    Shape squeezed_in_shape;
    Shape squeezed_out_shape;
    Squeeze(in_shape, &squeezed_in_shape, &in_squeezed_axis2original_axis);
    Squeeze(*out_shape, &squeezed_out_shape, &out_squeezed_axis2original_axis);
    GetGroupStartInAxis2OutAxis(squeezed_in_shape, squeezed_out_shape,
                                &squeezed_group_start_in_axis2out_axis);
  }
  for (const auto& pair : squeezed_group_start_in_axis2out_axis) {
    int64_t start_in_axis = in_squeezed_axis2original_axis.at(pair.first);
    int64_t start_out_axis = out_squeezed_axis2original_axis.at(pair.second);
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

REGISTER_OP(OperatorConf::kReshapeConf, ReshapeOp);

}  // namespace oneflow
