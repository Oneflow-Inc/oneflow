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

namespace oneflow {

namespace one {

namespace{

Maybe<void> JitInterpreter::ApplyImpl(const UserOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
    CHECK_EQ_OR_RETURN(inputs.size(), op_expr.input_size());
    //note
    if (inputs.size() == 0) {
        // NOTE(BBuf): handle for source UserOp like OFRecordReader, CoinFlip to MLIR.
        
        return Maybe<void>::Ok();
    }
    if (op_expr.op_type_name() == "copy") {
        // NOTE(BBuf): handle for copy UserOp which will NOT add op to MLIR.

        return Maybe<void>::Ok();
    }
    //   Normal UserOp inputs size >= 1 for infer parallel_desc.
    CHECK_GE_OR_RETURN(inputs.size(), 1);
    //update cached_op_expr_
    cached_op_expr_.push_back(*op_expr);
}

}

}

}