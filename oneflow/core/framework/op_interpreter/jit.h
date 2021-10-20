#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_JIT_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_JIT_H_
#include "oneflow/core/framework/op_expr.h"

namespace oneflow {

namespace one {

namespace ir {

class SimpleRuntime {
 public:
  SimpleRuntime() = default;
  SimpleRuntime(SimpleRuntime&&) = default;
  SimpleRuntime(const SimpleRuntime&) = default;
  SimpleRuntime& operator=(SimpleRuntime&&) = default;
  SimpleRuntime& operator=(const SimpleRuntime&) = default;
  ~SimpleRuntime() = default;
  void CacheOpExpr(const UserOpExpr& user_op_expr);
  std::function<void(const TensorTuple& inputs, TensorTuple* outputs)> ComplieCachedOpExpr(
      const UserOpExpr& user_op_expr);
};

}  // namespace ir

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_JIT_H_
