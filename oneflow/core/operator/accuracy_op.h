#ifndef ONEFLOW_CORE_OPERATOR_ACCURACY_OP_H_
#define ONEFLOW_CORE_OPERATOR_ACCURACY_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class AccuracyOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccuracyOp);
  AccuracyOp() = default;
  virtual ~AccuracyOp() = default;

  void InitFromOpConf() override;
  LogicalNode* NewProperLogicalNode() override { return new AccuracyLogicalNode; }

  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const;

 private:
  LogicalBlobId obn2lbi(const std::string& output_bn) const override {
    LogicalBlobId ret;
    ret.set_op_name(op_name());
    if (output_bn == "total_instance_num") {
      ret.set_blob_name("total_instance_num");
    } else {
      ret.set_blob_name(GetValFromCustomizedConf<std::string>(output_bn));
    }
    return ret;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ACCURACY_OP_H_
