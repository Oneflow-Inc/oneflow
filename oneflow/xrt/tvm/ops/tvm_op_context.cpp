#include "oneflow/xrt/tvm/ops/tvm_op_context.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

TVMOpContext::TVMOpContext(const XrtNode* node, tvm::Array<tvm::relay::Expr>&& node_inputs) 
  : node_(node), node_inputs_(std::move(node_inputs)) {}

void TVMOpContext::set_op_expr(tvm::relay::Expr op_expr) {
  op_expr_ = op_expr;
}

}
}
}
