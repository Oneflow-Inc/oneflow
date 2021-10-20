#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_JIT_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_JIT_H_
#include "oneflow/core/framework/op_expr.h"

namespace oneflow {

namespace one {

namespace ir {

class SimpleRuntime {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SimpleRuntime);
  SimpleRuntime() = default;
  ~SimpleRuntime() = default;
  virtual void CacheOpExpr(const UserOpExpr& user_op_expr) = 0;
  // TODO: should this function be wrapped in uniq ptr?
  virtual std::function<void(const TensorTuple& inputs, TensorTuple* outputs)> ComplieCachedOpExpr(
      const UserOpExpr& user_op_expr) = 0;
};

using InitRuntime = std::function<std::unique_ptr<SimpleRuntime>()>;
using RuntimeCreatorRegistry = HashMap<std::string, InitRuntime>;
RuntimeCreatorRegistry* GetRuntimeCreatorRegistry();
std::shared_ptr<SimpleRuntime> StartRuntime(const std::string& name);
void RegisterRuntimeCreator(const std::string& name, const InitRuntime& creator);

}  // namespace ir

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_JIT_H_
