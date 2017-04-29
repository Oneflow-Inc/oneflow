#ifndef ONEFLOW_OPERATOR_RELU_OP_H_
#define ONEFLOW_OPERATOR_RELU_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ReluOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluOp);
  ReluOp() = default;
  ~ReluOp() = default;

  std::string GetValueFromPbOpConf(const std::string& k) const override;
  void InitFromOpConf(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return true; }

  void InferShape4ObAndDtbFromIb() const override;
  void InferShape4ModelTmpBlob(ParallelPolicy policy,
                               uint64_t parallel_id) const override { }
  void InferShape4ModelDiffBlob(ParallelPolicy policy,
                                uint64_t parallel_id) const override { }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_RELU_OP_H_
