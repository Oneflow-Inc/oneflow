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

class TVMOpContext final : public OpContext {
 public:
  TVMOpContext(const XrtNode* node, 
      const PbMessage* message,
      util::Map<Argument, tvm::relay::Expr>&& input_arg2expr);
  ~TVMOpContext() = default;

  const XrtNode* node() const { return node_; }
  int32_t num_inputs() const { return input_name2arg_.size(); }

  void set_op_expr(tvm::relay::Expr op_expr);

  tvm::relay::Expr GetExpr4InputName(const std::string& name) const;
  const Shape& GetShape4InputName(const std::string& name) const;

  tvm::relay::Expr GetExpr4OutputName(const std::string& name) const;
  void SetExpr4OutputName(const std::string& name, tvm::relay::Expr&&);

 private:
  TVMOpContext(const TVMOpContext&) = delete;
  TVMOpContext& operator=(const TVMOpContext&) = delete;

  const XrtNode* node_;
  util::Map<std::string, tvm::relay::Expr> input_name2expr_;
  util::Map<std::string, Argument> input_name2arg_;
  util::Map<std::string, tvm::relay::Expr> output_name2expr_;
};

}
}
}

#endif // ONEFLOW_XRT_TVM_OPS_TVM_OP_CONTEXT_H_
