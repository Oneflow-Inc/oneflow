#ifndef OPERATOR_CONVOLUTION_OP_H_
#define OPERATOR_CONVOLUTION_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ConvolutionOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionOp);
  ConvolutionOp() = default;
  ~ConvolutionOp() = default;

  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return false; }

 private:

};

} // namespace oneflow

#endif // OPERATOR_CONVOLUTION_OP_H_
