#ifndef ONEFLOW_CORE_OPERATOR_BROADCAST_BINARY_OP_H_
#define ONEFLOW_CORE_OPERATOR_BROADCAST_BINARY_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BroadcastBinaryOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastBinaryOp);
  BroadcastBinaryOp() = default;
  ~BroadcastBinaryOp() override = default;

  void InitFromOpConf() override;
  bool IsAllOutputConst() const override { return GetValFromCustomizedConf<bool>("is_const"); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 protected:
  virtual Maybe<void> VirtualGetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const {
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BROADCAST_BINARY_OP_H_
