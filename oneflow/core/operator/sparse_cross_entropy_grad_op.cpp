#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SparseCrossEntropyGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseCrossEntropyGradOp);
  SparseCrossEntropyGradOp() = default;
  ~SparseCrossEntropyGradOp() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_sparse_cross_entropy_grad_conf());
    EnrollInputBn("prediction", false);
    EnrollInputBn("label");
    EnrollInputBn("dy");
    EnrollOutputBn("prediction_diff");
  }
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().sparse_cross_entropy_grad_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    *GetBlobDesc4BnInOp("prediction_diff") = *GetBlobDesc4BnInOp("prediction");
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
};

class SparseCrossEntropyMs1GradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseCrossEntropyMs1GradOp);
  SparseCrossEntropyMs1GradOp() = default;
  ~SparseCrossEntropyMs1GradOp() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_sparse_cross_entropy_ms1_grad_conf());
    EnrollInputBn("prediction", false);
    EnrollInputBn("label", false);
    EnrollInputBn("dy");
    EnrollOutputBn("prediction_diff");
  }
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().sparse_cross_entropy_ms1_grad_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    *GetBlobDesc4BnInOp("prediction_diff") = *GetBlobDesc4BnInOp("prediction");
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("prediction", 1)
        .Broadcast("dy")
        .Broadcast("label")
        .Split("prediction_diff", 1)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    const int64_t dim = op_conf().sparse_cross_entropy_ms1_grad_conf().depth();
    CHECK_GE(dim, parallel_ctx->parallel_num());
    BalancedSplitter bs(dim, parallel_ctx->parallel_num());
    kernel_conf->mutable_sparse_cross_entropy_grad_conf()->set_lower_bound(
        bs.At(parallel_ctx->parallel_id()).begin());
  }
};

REGISTER_OP(OperatorConf::kSparseCrossEntropyGradConf, SparseCrossEntropyGradOp);
REGISTER_OP(OperatorConf::kSparseCrossEntropyMs1GradConf, SparseCrossEntropyMs1GradOp);

}  // namespace oneflow
