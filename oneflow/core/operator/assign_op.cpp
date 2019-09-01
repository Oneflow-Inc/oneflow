#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AssignOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AssignOp);
  AssignOp() = default;
  ~AssignOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override {
    return NaiveInferHasBatchDim(HasBatchDim4BnInOp);
  }
  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void AssignOp::InitFromOpConf() {
  CHECK(op_conf().has_assign_conf());
  EnrollInputBn("ref")->set_is_mutable(true);
  EnrollInputBn("value");
}

const PbMessage& AssignOp::GetCustomizedConf() const { return op_conf().assign_conf(); }

Maybe<void> AssignOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("ref") == *GetBlobDesc4BnInOp("value"));
  return Maybe<void>::Ok();
}

void AssignOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(input_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kAssignConf, AssignOp);

}  // namespace oneflow
