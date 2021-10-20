#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/framework/op_interpreter/jit.h"

namespace oneflow {

namespace one {

namespace ir {

RuntimeCreatorRegistry* GetRuntimeRegistry() {
  static RuntimeCreatorRegistry registry;
  return &registry;
}

std::shared_ptr<SimpleRuntime> StartRuntime(const std::string& name) {
  static std::shared_ptr<SimpleRuntime> runtime;
  std::unique_ptr<SimpleRuntime> created = GetRuntimeRegistry()->at(name)();
  runtime = std::move(created);
  return runtime;
}

void RegisterRuntimeCreator(const std::string& name, const InitRuntime& creator) {
  if (GetRuntimeRegistry()->find(name) == GetRuntimeRegistry()->end()) {
    CHECK(GetRuntimeRegistry()->emplace(name, creator).second);
  }
}

}  // namespace ir

Maybe<void> JitInterpreter::Apply(const OpExpr& op_expr, const TensorTuple& inputs,
                                  TensorTuple* outputs, const OpExprInterpContext& ctx) const {
#define APPLY_IF(op_type)                                              \
  if (const auto* op = dynamic_cast<const op_type##Expr*>(&op_expr)) { \
    return ApplyImpl(*op, inputs, outputs, ctx);                       \
  }
  APPLY_IF(UserOp);
#undef APPLY_IF

  OF_UNIMPLEMENTED() << "The type " << op_expr.op_type_name()
                     << " has not been supported in LazyInterpreter::Apply.";
  return Maybe<void>::Ok();
}

Maybe<void> JitInterpreter::ApplyImpl(const UserOpExpr& op_expr, const TensorTuple& inputs,
                                      TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // update cached_op_expr_
  // TODO: MLIR add op expr

  return Maybe<void>::Ok();
}

}  // namespace one

}  // namespace oneflow
