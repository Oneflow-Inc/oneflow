#include "oneflow/core/operator/slice_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SliceOp::InitFromOpConf() {
  CHECK(op_conf().has_slice_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (op_conf().device_type() == DeviceType::kGPU) { EnrollConstBufBn("out_to_in_offset"); }
}

const PbMessage& SliceOp::GetCustomizedConf() const { return op_conf().slice_conf(); }

void SliceOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const Shape& in_shape = GetBlobDesc4BnInOp("in")->shape();
  in_shape.ToProto(kernel_conf->mutable_slice_conf()->mutable_in_shape());
}

Maybe<void> SliceOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx,
                                    const SbpSignature* sbp_signature,
                                    std::function<void(OpContext*)> EnrollOpCtx) const {
  auto ret = InferOutBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, sbp_signature, EnrollOpCtx);
  if (op_conf().device_type() == DeviceType::kGPU) {
    BlobDesc* offset_blob_desc = GetBlobDesc4BnInOp("out_to_in_offset");
    *offset_blob_desc = *GetBlobDesc4BnInOp("out");
    offset_blob_desc->set_data_type(DataType::kInt64);
  }
  return ret;
}

Maybe<void> SliceOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const SliceOpConf& conf = op_conf().slice_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ_OR_RETURN(conf.dim_slice_conf_size(), in_blob_desc->shape().NumAxes());
  DimVector shape_vec(in_blob_desc->shape().NumAxes());
  FOR_RANGE(size_t, i, 0, conf.dim_slice_conf_size()) {
    int32_t dim_len = in_blob_desc->shape().At(i);
    const DimSliceConf& dim_slice_conf = conf.dim_slice_conf(i);
    int32_t step = dim_slice_conf.stride();
    CHECK_GT_OR_RETURN(step, 0);
    int32_t start = dim_slice_conf.has_start() ? dim_slice_conf.start() : 0;
    int32_t end = dim_slice_conf.has_end() ? dim_slice_conf.end() : dim_len;
    if (start < 0) { start += dim_len; }
    if (end < 0) { end += dim_len; }
    if (end > dim_len) { end = dim_len; }
    CHECK_GE_OR_RETURN(start, 0);
    CHECK_LT_OR_RETURN(start, end);
    shape_vec[i] = (end - start - 1) / std::abs(step) + 1;
  }

  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape(shape_vec);
  return Maybe<void>::Ok();
}

Maybe<void> SliceOp::GetSbpSignatures(
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

REGISTER_OP(OperatorConf::kSliceConf, SliceOp);

}  // namespace oneflow
