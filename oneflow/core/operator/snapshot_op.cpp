#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class SnapshotOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SnapshotOp);
  SnapshotOp() = default;
  ~SnapshotOp() override = default;

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
  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override{};
};

void SnapshotOp::InitFromOpConf() {
  CHECK(op_conf().has_snapshot_conf());
  EnrollRepeatedInputBn("in", false);
}

const PbMessage& SnapshotOp::GetCustomizedConf() const { return op_conf().snapshot_conf(); }

REGISTER_CPU_OP(OperatorConf::kSnapshotConf, SnapshotOp);

}  // namespace oneflow
