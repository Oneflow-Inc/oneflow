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
                      const ParallelContext* parallel_ctx) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferOutputBlobSbpInferHint(
      std::function<SbpInferHint*(const std::string&)> SbpInferHint4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutputBlobSbpInferHint(SbpInferHint4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }

  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ACCURACY_OP_H_
