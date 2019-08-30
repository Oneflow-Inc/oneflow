#ifndef ONEFLOW_CORE_OPERATOR_LOSS_OP_H_
#define ONEFLOW_CORE_OPERATOR_LOSS_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class LossOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossOp);
  LossOp() = default;
  virtual ~LossOp() = default;

  void InitFromOpConf() override;
  LogicalNode* NewProperLogicalNode() const override { return new LossLogicalNode; }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  bool IsLossOp() const override { return true; }

 protected:
  virtual void VirtualInitFromOpConf() {}
  virtual Maybe<void> VirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {
    return Maybe<void>::Ok();
  }
  virtual LossKernelConf* GetMutLossKernelConf(KernelConf*) const = 0;

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  Maybe<void> InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SOFTMAX_LOSS_OP_H_
