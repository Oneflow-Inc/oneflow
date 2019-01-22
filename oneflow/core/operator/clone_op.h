#ifndef ONEFLOW_CORE_OPERATOR_CLONE_OP_H_
#define ONEFLOW_CORE_OPERATOR_CLONE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CloneOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CloneOp);
  CloneOp() = default;
  ~CloneOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferDiffBlobDescsWithoutFwBlob(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { UNIMPLEMENTED(); }
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }

  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return LogicalBlobId(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override { return LogicalBlobId(); }
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CLONE_OP_H_
