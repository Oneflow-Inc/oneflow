#include "oneflow/core/operator/operator.h"

namespace oneflow {

class FakeConsumeOp : public Operator {
 public:
  FakeConsumeOp() = default;
  virtual ~FakeConsumeOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    return Maybe<void>::Ok();
  };

 private:
  typedef std::function<OptInt64*(const std::string&)> BatchAxis4BnInOpFunc;
  Maybe<void> InferBatchAxis(
      BatchAxis4BnInOpFunc BatchAxis4BnInOp) const override {
    return Maybe<void>::Ok();
  };

  typedef std::function<Maybe<const BlobDesc*>(const std::string&)>
      LogicalBlobDesc4IbnFunc;
  Maybe<void> GetSbpSignatures(
      const LogicalBlobDesc4IbnFunc& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    return Maybe<void>::Ok();
  };
};

void FakeConsumeOp::InitFromOpConf() {
  CHECK(op_conf().has_fake_consume_conf());
  EnrollRepeatedInputBn("in", false);
}

const PbMessage& FakeConsumeOp::GetCustomizedConf() const {
  CHECK(op_conf().has_fake_consume_conf());
  return op_conf().fake_consume_conf();
}

REGISTER_OP(OperatorConf::kFakeConsumeConf, FakeConsumeOp);

}  // namespace oneflow
