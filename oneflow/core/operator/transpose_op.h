#ifndef ONEFLOW_CORE_OPERATOR_TRANSPOSE_OP_H_
#define ONEFLOW_CORE_OPERATOR_TRANSPOSE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class TransposeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TransposeOp);
  TransposeOp() = default;
  ~TransposeOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  void InferOutBlobModelSplitAxis(std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override;

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_TRANSPOSE_OP_H_
