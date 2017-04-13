#ifndef ONEFLOW_OPERATOR_SPLIT_OP_H_
#define ONEFLOW_OPERATOR_SPLIT_OP_H_

#include "operator/operator.h"

namespace oneflow {

class SplitOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SplitOp);
  SplitOp() = default;
  ~SplitOp() = default;

  void Init(const OperatorConf& op_conf) override;
  void InferBlobDesc4ObAndDtbFromIb() const override { TODO(); }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_SPLIT_OP_H_
