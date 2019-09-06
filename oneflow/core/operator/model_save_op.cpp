#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class ModelSaveOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveOp);
  ModelSaveOp() = default;
  ~ModelSaveOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  LogicalNode* NewProperLogicalNode() const override { return new PrintLogicalNode; }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return Maybe<void>::Ok();
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    return Maybe<void>::Ok();
  };
};

void ModelSaveOp::InitFromOpConf() {
  CHECK(op_conf().has_model_save_conf());
  EnrollInputBn("path", false);
  EnrollRepeatedInputBn("in", false);
}

const PbMessage& ModelSaveOp::GetCustomizedConf() const { return op_conf().model_save_conf(); }

REGISTER_CPU_OP(OperatorConf::kModelSaveConf, ModelSaveOp);

}  // namespace oneflow
