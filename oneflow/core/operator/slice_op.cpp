#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class SliceOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceOp);
  SliceOp() = default;
  ~SliceOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_slice_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out");
    if (op_conf().device_type() == DeviceType::kGPU) { EnrollConstBufBn("out_to_in_offset"); }
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().slice_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    auto ret = InferOutBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, sbp_signature, EnrollOpCtx);
    if (op_conf().device_type() == DeviceType::kGPU) {
      BlobDesc* offset_blob_desc = GetBlobDesc4BnInOp("out_to_in_offset");
      *offset_blob_desc = *GetBlobDesc4BnInOp("out");
      offset_blob_desc->set_data_type(DataType::kInt64);
    }
    return ret;
  }
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature,
                                std::function<void(OpContext*)> EnrollOpCtx) const override {
    const SliceOpConf& conf = op_conf().slice_conf();
    const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
    CHECK_EQ_OR_RETURN(conf.dim_slice_conf_size(), in_blob_desc->shape().NumAxes());
    std::vector<int64_t> shape_vec(in_blob_desc->shape().NumAxes());
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
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    const Shape& in_shape = GetBlobDesc4BnInOp("in")->shape();
    in_shape.ToProto(kernel_conf->mutable_slice_conf()->mutable_in_shape());
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
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
};

class SliceGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceGradOp);
  SliceGradOp() = default;
  ~SliceGradOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_slice_grad_conf());
    EnrollInputBn("dy", false);
    EnrollInputBn("like", false)->set_use_header_only(true);
    EnrollOutputBn("dx", false);
    if (op_conf().device_type() == DeviceType::kGPU) { EnrollConstBufBn("y_to_x_offset"); }
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().slice_grad_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
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
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    const Shape& in_shape = GetBlobDesc4BnInOp("like")->shape();
    in_shape.ToProto(kernel_conf->mutable_slice_conf()->mutable_in_shape());
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
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
};

REGISTER_OP(OperatorConf::kSliceConf, SliceOp);
REGISTER_OP(OperatorConf::kSliceGradConf, SliceGradOp);

}  // namespace oneflow
