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
  LogicalNode* NewProperLogicalNode() override { return new LossLogicalNode; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;
  bool IsLossOp() const override { return true; }

 protected:
  virtual void VirtualInitFromOpConf() {}
  virtual void VirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {}
  virtual LossKernelConf* GetMutLossKernelConf(KernelConf*) const = 0;

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  LogicalBlobId obn2lbi(const std::string& output_bn) const override {
    LogicalBlobId ret;
    ret.set_op_name(op_name());
    if (output_bn == "total_instance_num") {
      ret.set_blob_name("total_instance_num");
    } else if (output_bn == "reduction_coefficient") {
      ret.set_blob_name("reduction_coefficient");
    } else {
      ret.set_blob_name(GetValFromCustomizedConf<std::string>(output_bn));
    }
    return ret;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SOFTMAX_LOSS_OP_H_
