#include "oneflow/core/operator/operator.h"

namespace oneflow {

class L2NormalizeGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L2NormalizeGradOp);
  L2NormalizeGradOp() = default;
  ~L2NormalizeGradOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_l2_normalize_grad_conf());
    EnrollInputBn("dy");
    EnrollInputBn("y");
    EnrollInputBn("square_x_sum");
    EnrollOutputBn("dx");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().l2_normalize_grad_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const L2NormalizeGradOpConf& conf = op_conf().l2_normalize_grad_conf();
    const BlobDesc* dy_blob_desc = GetBlobDesc4BnInOp("dy");
    CHECK_GE_OR_RETURN(conf.axis(), 0);
    CHECK_LT_OR_RETURN(conf.axis(), dy_blob_desc->shape().NumAxes());
    CHECK_GT_OR_RETURN(conf.epsilon(), 0);
    *GetBlobDesc4BnInOp("dx") = *dy_blob_desc;
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kL2NormalizeGradConf, L2NormalizeGradOp);

}  // namespace oneflow
