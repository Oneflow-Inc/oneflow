#ifndef ONEFLOW_CORE_OPERATOR_SLICE_OP_H_
#define ONEFLOW_CORE_OPERATOR_SLICE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SliceOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceOp);
  SliceOp() = default;
  ~SliceOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool IsInputBnInOpAllowedModelSplit(const std::string& ibn) const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutputBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp,
                                       parallel_context);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SLICE_OP_H_
