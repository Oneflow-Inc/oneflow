#ifndef ONEFLOW_CORE_OPERATOR_SCALAR_ADD_OP_H_
#define ONEFLOW_CORE_OPERATOR_SCALAR_ADD_OP_H_

#include "oneflow/core/operator/cwise_op.h"

namespace oneflow {

class ScalarAddOp final : public CWiseOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarAddOp);
  ScalarAddOp() = default;
  ~ScalarAddOp() = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().scalar_add_conf(); }
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SCALAR_ADD_OP_H_
