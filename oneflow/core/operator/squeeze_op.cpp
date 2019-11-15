#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SqueezeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SqueezeOp);
  SqueezeOp() = default;
  ~SqueezeOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().squeeze_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void SqueezeOp::InitFromOpConf() {
  CHECK(op_conf().has_squeeze_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

Maybe<void> SqueezeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  DimVector dim_vec = in->shape().dim_vec();
  for (const auto& idx : PbRf2StdVec(op_conf().squeeze_conf().axis())) {
    CHECK_LT_OR_RETURN(idx, dim_vec.size());
    CHECK_EQ_OR_RETURN(dim_vec[idx], 1);
    dim_vec[idx] = -1;
  }
  dim_vec.erase(std::remove(dim_vec.begin(), dim_vec.end(), -1), dim_vec.end());
  out->mut_shape() = Shape(dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> SqueezeOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  auto squeeze_axes_vec = PbRf2StdVec(op_conf().squeeze_conf().axis());
  auto squeeze_dim0_iter = std::find(squeeze_axes_vec.begin(), squeeze_axes_vec.end(), 0);
  if (squeeze_dim0_iter == squeeze_axes_vec.end()) {
    SbpSignatureBuilder().Split("in", 0).Split("out", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
  } else {
    SbpSignatureBuilder().Split("in", 0).Broadcast("out").Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kSqueezeConf, SqueezeOp);

}  // namespace oneflow
