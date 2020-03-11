#ifndef ONEFLOW_XRT_TVM_OPS_TVM_OP_CONTEXT_H_
#define ONEFLOW_XRT_TVM_OPS_TVM_OP_CONTEXT_H_

#include "oneflow/xrt/argument.h"
#include "oneflow/xrt/kernel/op_context.h"
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include "oneflow/xrt/graph/node.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

class TVMOpContext final {
 public:
  TVMOpContext(const XrtNode* node, tvm::Array<tvm::relay::Expr>&& node_inputs);
  ~TVMOpContext() = default;

  const XrtNode* node() const { return node_; }
  const tvm::Array<tvm::relay::Expr> node_inputs() const { return node_inputs_; }
  tvm::relay::Expr op_expr() const { return op_expr_; }

  void set_op_expr(tvm::relay::Expr op_expr);

 private:
  TVMOpContext(const TVMOpContext&) = delete;
  TVMOpContext& operator=(const TVMOpContext&) = delete;

  const XrtNode* node_;
  tvm::Array<tvm::relay::Expr> node_inputs_;
  tvm::relay::Expr op_expr_;
};

}
}
}

#endif // ONEFLOW_XRT_TVM_OPS_TVM_OP_CONTEXT_H_
