#ifndef ONEFLOW_OPERATOR_COPY_OP_H_
#define ONEFLOW_OPERATOR_COPY_OP_H_

#include "operator/operator.h"

namespace oneflow {

class CopyOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyOp);
  CopyOp() = default;
  ~CopyOp() = default;

  void Init(const OperatorConf& op_conf) override;

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_COPY_OP_H_
