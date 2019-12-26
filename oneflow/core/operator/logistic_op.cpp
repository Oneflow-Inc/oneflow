#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LogisticOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogisticOp);
  LogisticOp() = default;
  ~LogisticOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_logistic_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().logistic_conf(); }

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

REGISTER_OP(OperatorConf::kLogisticConf, LogisticOp);

}  // namespace oneflow
