#include "oneflow/xrt/tvm/ops/tvm_op_context.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

TVMOpContext::TVMOpContext(const XrtNode* node, 
    const PbMessage* message,
    util::Map<Argument, tvm::relay::Expr>&& input_arg2expr) :
  OpContext(*message), node_(node), input_name2expr_(), input_name2arg_(), output_name2expr_() {
  for (const auto& pair : input_arg2expr) {
    std::string input_name = pair.first.meta_data().consume_key;
    input_name2expr_.emplace(input_name, pair.second);
    input_name2arg_.emplace(input_name, pair.first);
  }
}

tvm::relay::Expr TVMOpContext::GetExpr4InputName(const std::string& name) const {
  auto it = input_name2expr_.find(name);
  CHECK(it != input_name2expr_.end())
    << "Cannot find input_name: " << name << " in TVMOpContext of node: " << node_->name();
  return it->second;
}

const Shape& TVMOpContext::GetShape4InputName(const std::string& name) const {
  auto it = input_name2arg_.find(name);
  CHECK(it != input_name2arg_.end())
    << "Cannot find input_name: " << name << " in TVMOpContext of node: " << node_->name();
  return it->second.shape();
}

tvm::relay::Expr TVMOpContext::GetExpr4OutputName(const std::string& name) const {
  auto it = output_name2expr_.find(name);
  CHECK(it != output_name2expr_.end())
    << "Cannot find output_name: " << name << " in TVMOpContext of node: " << node_->name();
  return it->second;
}

void TVMOpContext::SetExpr4OutputName(const std::string& name, tvm::relay::Expr&& expr) {
  CHECK(output_name2expr_.emplace(name, std::move(expr)).second);
}

}
}
}
