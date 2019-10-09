#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SleepOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SleepOp);
  SleepOp() = default;
  ~SleepOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void SleepOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_const_inplace_ibn("in");
}

Maybe<void> SleepOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

const PbMessage& SleepOp::GetCustomizedConf() const { return op_conf().sleep_conf(); }

Maybe<void> SleepOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  return NaiveInferBatchAxis(BatchAxis4BnInOp);
}

Maybe<void> SleepOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const auto bns = StdVec2PbRpf<std::string>({"in", "out"});
  SbpSignatureBuilder().PartialSum(bns).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  const int64_t num_axes = JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes();
  SbpSignatureBuilder().Split(bns, 0).MakeSplitSignatureListBuilder(num_axes).Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kSleepConf, SleepOp);

}  // namespace oneflow
