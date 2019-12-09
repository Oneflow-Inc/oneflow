#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

int64_t CheckGatherAxis(const LocalGatherOpConf& conf, const int64_t num_axes) {
  const int64_t axis = conf.axis();
  CHECK_GE(axis, 0);
  CHECK_LT(axis, num_axes);
  return axis;
}

}  // namespace

class LocalGatherOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalGatherOp);
  LocalGatherOp() = default;
  ~LocalGatherOp() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_local_gather_conf());
    EnrollInputBn("indices", false);
    EnrollInputBn("in");
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().local_gather_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
    CHECK_OR_RETURN(IsIntegralDataType(indices->data_type()));
    CHECK_GT_OR_RETURN(indices->shape().NumAxes(), 0);
    CHECK_EQ_OR_RETURN(indices->is_tensor_list(), false);
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    CHECK_GT_OR_RETURN(in->shape().NumAxes(), 0);
    const int64_t axis = CheckGatherAxis(op_conf().local_gather_conf(), in->shape().NumAxes());
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    DimVector dim_vec;
    dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin(),
                   in->shape().dim_vec().cbegin() + axis);
    dim_vec.insert(dim_vec.end(), indices->shape().dim_vec().cbegin(),
                   indices->shape().dim_vec().cend());
    dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin() + axis + 1,
                   in->shape().dim_vec().end());
    out->mut_shape() = Shape(dim_vec);
    out->set_is_dynamic(in->is_dynamic() || indices->is_dynamic());
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    const int64_t axis =
        CheckGatherAxis(op_conf().local_gather_conf(), GetBlobDesc4BnInOp("in")->shape().NumAxes());
    kernel_conf->mutable_gather_conf()->set_axis(axis);
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    SbpSignatureList sbp_sig_list;
    JUST(GetSbpSignatures(&sbp_sig_list));
    CHECK_EQ(sbp_sig_list.sbp_signature_size(), 1);
    *sbp_signature = sbp_sig_list.sbp_signature(0);
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("indices", 0)
        .Split("in", 0)
        .Split("out", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

class LocalGatherGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalGatherGradOp);
  LocalGatherGradOp() = default;
  ~LocalGatherGradOp() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_local_gather_grad_conf());
    EnrollInputBn("segment_ids", false);
    EnrollInputBn("like", false)->set_use_header_only(true);
    EnrollInputBn("data");
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().local_gather_grad_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* segment_ids = GetBlobDesc4BnInOp("segment_ids");
    CHECK_OR_RETURN(IsIntegralDataType(segment_ids->data_type()));
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->CopyMetaFrom(*GetBlobDesc4BnInOp("like"));
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("segment_ids", 0)
        .Split("data", 0)
        .Split("out", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }

  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    SbpSignatureList sbp_sig_list;
    JUST(GetSbpSignatures(&sbp_sig_list));
    CHECK_EQ(sbp_sig_list.sbp_signature_size(), 1);
    *sbp_signature = sbp_sig_list.sbp_signature(0);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kLocalGatherConf, LocalGatherOp);
REGISTER_OP(OperatorConf::kLocalGatherGradConf, LocalGatherGradOp);

}  // namespace oneflow
