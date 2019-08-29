#ifndef ONEFLOW_CORE_OPERATOR_CALLBACK_NOTIFY_OP_H_
#define ONEFLOW_CORE_OPERATOR_CALLBACK_NOTIFY_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CallbackNotifyOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CallbackNotifyOp);
  CallbackNotifyOp() = default;
  ~CallbackNotifyOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  LogicalNode* NewProperLogicalNode() const override;

 private:
  Maybe<void> InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CALLBACK_NOTIFY_OP_H_
