#include "oneflow/core/operator/slice_grad_op.h"
#include "oneflow/core/job/sbp_signature_rule.h"

namespace oneflow {

namespace {

class SliceGradPartialSumSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceGradPartialSumSignatureRule);
  ~SliceGradPartialSumSignatureRule() override = default;

  SliceGradPartialSumSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": P -> P"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const auto& dy_sbp_infer_hint = SbpInferHint4Ibn("dy");
    if (!dy_sbp_infer_hint.sbp_parallel().has_partial_sum_parallel()) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    if (parallel_desc.parallel_num() != dy_sbp_infer_hint.parallel_num()) {
      return MakeSbpSigMatchParallelNumError(parallel_desc.parallel_num(),
                                             dy_sbp_infer_hint.parallel_num());
    }
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*bn2sbp)["like"] = SbpInferHint4Ibn("like").sbp_parallel();
    (*bn2sbp)["dy"].mutable_partial_sum_parallel();
    (*bn2sbp)["dx"].mutable_partial_sum_parallel();
  }
};

}  // namespace

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

void SliceGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  const SliceGradOpConf& conf = op_conf().slice_grad_conf();
  const BlobDesc* like_blob_desc = GetBlobDesc4BnInOp("like");
  CHECK_EQ(conf.dim_slice_conf_size(), like_blob_desc->shape().NumAxes() - 1);
  GetBlobDesc4BnInOp("dx")->CopyMetaFrom(*like_blob_desc);
  if (op_conf().device_type() == DeviceType::kGPU) {
    BlobDesc* offset_blob_desc = GetBlobDesc4BnInOp("y_to_x_offset");
    *offset_blob_desc = *GetBlobDesc4BnInOp("dy");
    offset_blob_desc->set_data_type(DataType::kInt64);
  }
}

void SliceGradOp::GetSbpSignatureRules(
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new SliceGradPartialSumSignatureRule(this));
  rules->emplace_back(MakeDataSplitSbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kSliceGradConf, SliceGradOp);

}  // namespace oneflow
