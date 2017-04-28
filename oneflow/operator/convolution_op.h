#ifndef OPERATOR_CONVOLUTION_OP_H_
#define OPERATOR_CONVOLUTION_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ConvolutionOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionOp);
  ConvolutionOp() = default;
  ~ConvolutionOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  void InferShape4ObAndDtbFromIb() const override { TODO(); }
  std::string GetValueFromPbOpConf(const std::string& k) const override;
  void InferShape4ModelTmpBlob(ParallelPolicy policy,
                               uint64_t parallel_id) const override {
    TODO();
  }
  void InferShape4ModelDiffBlob(ParallelPolicy policy,
                                uint64_t parallel_id) const override {
    TODO();
  }

 private:

};

} // namespace oneflow

#endif // OPERATOR_CONVOLUTION_OP_H_
