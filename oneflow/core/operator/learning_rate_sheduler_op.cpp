#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LearningRateShedulerOp : public Operator {
 public:
  LearningRateShedulerOp() = default;
  virtual ~LearningRateShedulerOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

  LogicalNode* NewProperLogicalNode() const override {
    return new RecordLoadLogicalNode;
  }

 private:
  typedef std::function<bool*(const std::string&)> HasBatchDim4BnInOpFunc;
  void InferHasBatchDim(
      HasBatchDim4BnInOpFunc HasBatchDim4BnInOp) const override {
    *HasBatchDim4BnInOp("out") = false;
  }

  typedef std::function<const BlobDesc&(const std::string&)>
      LogicalBlobDesc4IbnFunc;
  void GetSbpSignatures(const LogicalBlobDesc4IbnFunc& LogicalBlobDesc4Ibn,
                        SbpSignatureList* sbp_sig_list) const override;
};

void LearningRateShedulerOp::InitFromOpConf() {
  CHECK(op_conf().has_lr_sheduler_conf());
  EnrollOutputBn("out", false);
}

const PbMessage& LearningRateShedulerOp::GetCustomizedConf() const {
  CHECK(op_conf().has_lr_sheduler_conf());
  return op_conf().lr_sheduler_conf();
}

void LearningRateShedulerOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc *out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape({1});
  out_blob_desc->set_data_type(oneflow::kFloat);
}

typedef std::function<const BlobDesc&(const std::string&)>
      LogicalBlobDesc4IbnFunc;
void LearningRateShedulerOp::GetSbpSignatures(
    const LogicalBlobDesc4IbnFunc& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast("out")
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kLrShedulerConf, LearningRateShedulerOp);

}  // namespace oneflow
