#ifndef ONEFLOW_CORE_OPERATOR_CONVOLUTION_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONVOLUTION_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class ConvolutionOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionOp);
  ConvolutionOp() = default;
  ~ConvolutionOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx);

  void VirtualFixParallelDesc(ParallelDesc* pr_desc) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONVOLUTION_OP_H_
