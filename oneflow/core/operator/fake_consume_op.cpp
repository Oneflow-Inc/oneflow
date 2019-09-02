#include "oneflow/core/operator/operator.h"

namespace oneflow {

class FakeConsumeOp : public Operator {
 public:
  FakeConsumeOp() = default;
  virtual ~FakeConsumeOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override {};

 private:
  typedef std::function<bool*(const std::string&)> HasBatchDim4BnInOpFunc;
  void InferHasBatchDim(
      HasBatchDim4BnInOpFunc HasBatchDim4BnInOp) const override {};

  typedef std::function<const BlobDesc&(const std::string&)>
      LogicalBlobDesc4IbnFunc;
  void GetSbpSignatures(const LogicalBlobDesc4IbnFunc& LogicalBlobDesc4Ibn,
                        SbpSignatureList* sbp_sig_list) const override {};
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
