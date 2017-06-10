#ifndef ONEFLOW_CORE_OPERATOR_CLEAR_OP_H_
#define ONEFLOW_CORE_OPERATOR_CLEAR_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/register_desc.h"

// TODO() : using clear op to assist model diff acc

namespace oneflow {

class ClearOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClearOp);
  ClearOp() = default;
  ~ClearOp() = default;
  
 private:

};

} // namespace oneflow

#endif // ONEFLOW_CORE_OPERATOR_CLEAR_OP_H_
