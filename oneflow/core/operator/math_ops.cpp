#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ScalarMulOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarMulOp);
  ScalarMulOp() = default;
  ~ScalarMulOp() = default;

  void InitFromOpConf() override {
    EnrollInputBn("in");
    EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
    return Maybe<void>::Ok();
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().scalar_mul_conf(); }

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
        .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes())
        .Build(sbp_sig_list);
    SbpSignatureBuilder()
        .PartialSum(input_bns())
        .PartialSum(output_bns())
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kScalarMulConf, ScalarMulOp);

class ScalarAddOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarAddOp);
  ScalarAddOp() = default;
  ~ScalarAddOp() = default;

  void InitFromOpConf() override {
    EnrollInputBn("in");
    EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
    return Maybe<void>::Ok();
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().scalar_add_conf(); }

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
        .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes())
        .Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kScalarAddConf, ScalarAddOp);

}  // namespace oneflow
