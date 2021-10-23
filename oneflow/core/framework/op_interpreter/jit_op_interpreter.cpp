#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/framework/op_interpreter/jit_op_interpreter.h"
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

namespace oneflow {

namespace one {

using namespace mlir;

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

std::string GetDeviceTag(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_cuda()) {
    return "gpu";
  } else {
    return "cpu";
  }
}

Maybe<const ParallelDesc> GetParallelDesc(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_local()) {
    const auto& device = JUST(tensor->device());
    const auto& placement = JUST(Placement4Device(device));
    return placement.shared_from_symbol();
  } else {
    return JUST(tensor->parallel_desc()).shared_from_symbol();
  }
}

Maybe<void> JitInterpreter::ApplyImpl(const UserOpExpr& op_expr, const TensorTuple& inputs,
                                      TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  LOG(ERROR) << "[adding] " << op_expr.proto().DebugString();
  auto op_conf = JUST(OpInterpUtil::GenBuiltinOpConf(op_expr, ctx.attrs));
  LOG(ERROR) << "[op_conf] " << op_conf->DebugString();
  const std::string device_tag = GetDeviceTag(inputs.at(0));
  const bool is_local = inputs.at(0)->is_local();
  const std::shared_ptr<const ParallelDesc> parallel_desc = JUST(GetParallelDesc(inputs.at(0)));
  op_conf->set_device_tag(device_tag);

  for (int i = 0; i < inputs.size(); ++i) {
    const auto& input_tensor = inputs.at(i);
    CHECK_OR_RETURN(device_tag == GetDeviceTag(input_tensor));
    CHECK_OR_RETURN(parallel_desc->EqualsIgnoringHierarchy(*JUST(GetParallelDesc(input_tensor))));
    CHECK_EQ_OR_RETURN(is_local, input_tensor->is_local());
  }
  CHECK_EQ_OR_RETURN(outputs->size(), op_expr.output_size());
  auto op = JUST(ConstructOp(*op_conf));
  std::vector<NamedAttribute> attr_vec;
  std::vector<::mlir::Value> operand_vec;
  // JUST(op->InferLogicalOutBlobDescs(
  //     [](const std::string& bn_in_op) {
  //       LOG(ERROR) << "bn_in_op: " << bn_in_op;
  //       auto bd = new BlobDesc(DataType::kInvalidDataType);
  //       return bd;
  //     },
  //     *parallel_desc));
  auto indexed_arg_name_and_index = op_expr.input_arg_tuple()->indexed_arg_name_and_index();
  CHECK_EQ_OR_RETURN(indexed_arg_name_and_index.size(), inputs.size());
  importer_.GetOrInsertFuncAndCreateMapping(GetJitFuncName(), indexed_arg_name_and_index, inputs,
                                            outputs);
  CHECK_OR_RETURN(importer_.ProcessUserOp(*op_conf).succeeded());
  LOG(ERROR) << "[func name] " << GetJitFuncName();
  LOG(ERROR) << "[applied] " << op->op_name();
  module_->dump();
  // TODO: MLIR add op expr
  return Maybe<void>::Ok();
}

}  // namespace one

}  // namespace oneflow
