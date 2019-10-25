#include "oneflow/core/operator/slice_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SliceGradOp::InitFromOpConf() {
  CHECK(op_conf().has_slice_grad_conf());
  EnrollInputBn("dy", false);
  EnrollInputBn("like", false)->set_use_header_only(true);
  EnrollOutputBn("dx", false);
  if (op_conf().device_type() == DeviceType::kGPU) { EnrollConstBufBn("y_to_x_offset"); }
}

const PbMessage& SliceGradOp::GetCustomizedConf() const { return op_conf().slice_grad_conf(); }

void SliceGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const Shape& in_shape = GetBlobDesc4BnInOp("like")->shape();
  in_shape.ToProto(kernel_conf->mutable_slice_conf()->mutable_in_shape());
}

Maybe<void> SliceGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const SliceGradOpConf& conf = op_conf().slice_grad_conf();
  const BlobDesc* like_blob_desc = GetBlobDesc4BnInOp("like");
  CHECK_EQ_OR_RETURN(conf.dim_slice_conf_size(), like_blob_desc->shape().NumAxes());
  GetBlobDesc4BnInOp("dx")->CopyMetaFrom(*like_blob_desc);
  if (op_conf().device_type() == DeviceType::kGPU) {
    BlobDesc* offset_blob_desc = GetBlobDesc4BnInOp("y_to_x_offset");
    *offset_blob_desc = *GetBlobDesc4BnInOp("dy");
    offset_blob_desc->set_data_type(DataType::kInt64);
  }
  return Maybe<void>::Ok();
}

Maybe<void> SliceGradOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder()
      .PartialSum(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kSliceGradConf, SliceGradOp);

}  // namespace oneflow
