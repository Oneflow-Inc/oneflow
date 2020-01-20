#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RsqrtOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RsqrtOp);
  RsqrtOp() = default;
  ~RsqrtOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_rsqrt_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().rsqrt_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
    return Maybe<void>::Ok();
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
        .MakeSplitSignatureListBuilder(
            JUST(LogicalBlobDesc4Ibn(input_bns().Get(0)))->shape().NumAxes())
        .Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kRsqrtConf, RsqrtOp);

}  // namespace oneflow
