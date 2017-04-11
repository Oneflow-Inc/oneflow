#ifndef ONEFLOW_OPERATOR_CONCAT_OP_H_
#define ONEFLOW_OPERATOR_CONCAT_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ConcatOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConcatOp);
  ConcatOp() = default;
  ~ConcatOp() = default;

  void Init(const OperatorConf& op_conf) override;
  void InferBlobDesc4ObAndDtbFromIb() const override { TODO(); }
  void InferBlobDesc4MbAndMtb() const override { TODO(); }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_CONCAT_OP_H_
