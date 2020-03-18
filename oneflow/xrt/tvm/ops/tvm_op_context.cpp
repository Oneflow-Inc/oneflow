#include "oneflow/xrt/tvm/ops/tvm_op_context.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

void TVMOpContext::set_op_expr(tvm::relay::Expr op_expr) {
  op_expr_ = op_expr;
}

tvm::relay::Expr TVMOpContext::GetExpr4InputName(const std::string& name) {
  auto it = input_name2expr_.find(name);
  CHECK(it != input_name2expr_.end());
  return it->second;
}

}
}
}
