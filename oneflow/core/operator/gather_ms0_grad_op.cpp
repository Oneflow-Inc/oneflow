#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

class GatherMs0GradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherMs0GradOp);
  GatherMs0GradOp() = default;
  ~GatherMs0GradOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
};

void GatherMs0GradOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_ms0_grad_conf());
  EnrollInputBn("indices", false);
  EnrollInputBn("out_diff", false);
  EnrollOutputBn("in_diff", false);
}

const PbMessage& GatherMs0GradOp::GetCustomizedConf() const {
  return op_conf().gather_ms0_grad_conf();
}

Maybe<void> GatherMs0GradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const GatherMs0GradOpConf& conf = op_conf().gather_ms0_grad_conf();
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK_OR_RETURN(IsIndexDataType(indices->data_type()));
  const BlobDesc* out_diff = GetBlobDesc4BnInOp("out_diff");
  std::vector<int64_t> in_diff_dim_vec;
  BalancedSplitter bs(conf.gather_dim_size(), parallel_ctx->parallel_num());
  in_diff_dim_vec.push_back(bs.At(parallel_ctx->parallel_id()).size());
  in_diff_dim_vec.insert(in_diff_dim_vec.end(),
                         out_diff->shape().dim_vec().cbegin() + indices->shape().NumAxes(),
                         out_diff->shape().dim_vec().end());
  BlobDesc* in_diff = GetBlobDesc4BnInOp("in_diff");
  in_diff->set_data_type(out_diff->data_type());
  in_diff->mut_shape() = Shape(in_diff_dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> GatherMs0GradOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  SbpSignatureList sbp_sig_list;
  JUST(GetSbpSignatures(&sbp_sig_list));
  CHECK_EQ(sbp_sig_list.sbp_signature_size(), 1);
  *sbp_signature = sbp_sig_list.sbp_signature(0);
  return Maybe<void>::Ok();
}

Maybe<void> GatherMs0GradOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast("indices")
      .Broadcast("out_diff")
      .Split("in_diff", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

Maybe<void> GatherMs0GradOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("in_diff")->clear_value();
  return Maybe<void>::Ok();
}

void GatherMs0GradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const GatherMs0GradOpConf& conf = op_conf().gather_ms0_grad_conf();
  BalancedSplitter bs(conf.gather_dim_size(), parallel_ctx->parallel_num());
  int64_t offset = bs.At(parallel_ctx->parallel_id()).begin();
  kernel_conf->mutable_gather_ms0_grad_conf()->set_offset(offset);
}

REGISTER_OP(OperatorConf::kGatherMs0GradConf, GatherMs0GradOp);

}  // namespace oneflow
