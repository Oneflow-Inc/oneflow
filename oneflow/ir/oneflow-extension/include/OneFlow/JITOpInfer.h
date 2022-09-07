#include "oneflow/core/framework/infer_util.h"

namespace oneflow {

namespace ir {

namespace jit {

Maybe<void> InferTensorDesc(user_op::InferContext* ctx);

}  // namespace jit

}  // namespace ir

}  // namespace oneflow
