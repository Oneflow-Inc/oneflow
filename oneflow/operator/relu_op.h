#ifndef ONEFLOW_OPERATOR_RELU_OP_H_
#define ONEFLOW_OPERATOR_RELU_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ReluOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluOp);
  ReluOp() = default;
  ~ReluOp() = default;

  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return true; }

  void InferShape4ObAndDtbFromIb() const override { TODO(); }
  void InferShape4Mtb(ParallelPolicy, uint64_t parallel_id) const override {
    TODO();
  }
  void InferShape4Mdb(ParallelPolicy, uint64_t parallel_id) const override {
    TODO();
  }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_RELU_OP_H_
