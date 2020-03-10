#include "oneflow/xrt/tvm/tvm_graph_compiler.h"
#include "oneflow/xrt/tvm/tvm_executable.h"
#include <tvm/runtime/device_api.h>

namespace oneflow {
namespace xrt {

namespace {

tvm::Array<tvm::relay::IndexExpr> ConvertShapeToTVM(const oneflow::Shape& shape) {
  tvm::Array<tvm::relay::IndexExpr> ret;
  for (int i = 0;i < shape.NumAxes(); ++i) {
    ret.push_back(tvm::relay::IndexExpr(static_cast<int32_t>(shape.At(i))));
  }
  return ret;
}

tvm::relay::DataType ConvertDataTypeToTVM(DataType dtype) {
  static const util::Map<DataType, tvm::relay::DataType> type_map = {
    {DataType::kChar, tvm::Int(8)},
    {DataType::kFloat, tvm::Float(16)},
    {DataType::kDouble, tvm::Float(32)},
    {DataType::kInt8, tvm::Int(8)},
    {DataType::kInt32, tvm::Int(32)},
    {DataType::kInt64, tvm::Float(64)},
    {DataType::kUInt8, tvm::UInt(8)},
    {DataType::kFloat16, tvm::Float(16)},
  };
  CHECK(type_map.find(dtype) != type_map.end());
  return type_map.at(dtype);
}

}

TVMGraphCompiler::TVMGraphCompiler(const std::string& name) : GraphCompiler::Impl(name) {
  builder_ = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
}

std::shared_ptr<Executable> Compile(const XrtGraph *graph,
                                    const std::vector<Parameter> &entry_params,
                                    const std::vector<Parameter> &return_params,
                                    const std::vector<InputOutputAlias> &aliases) {
  util::Map<std::string, tvm::relay::Expr> tensor_name2expr;
  tvm::Array<tvm::relay::Var> input_vars;

  for (const auto& para : entry_params) {
    auto tensor_type = tvm::relay::TensorTypeNode::make(
        ConvertShapeToTVM(para.shape()),
        ConvertDataTypeToTVM(para.data_type()));
    auto var = tvm::relay::VarNode::make(input_arg_name,
        tensor_type);
    tensor_name2expr.emplace(para.name(), var);
    input_vars.push_back(var);
  }

  algorithm::TopologyVisit(*graph, [&](const XrtNode* node) {
    if (node->IsArgumentNode()) { continue; }
    tvm::Array<tvm::relay::Expr> tvm_inputs;
    for (const auto* in_edge : node->in_edges()) {
      auto it = tensor_name2expr.find(in_edge->argument().name());
      CHECK(it != tensor_name2expr.end());
      tvm_inputs.push_back(it->second);
    }
    auto expr = ConvertOpNodeToTVM(node, tvm_inputs);
    for (const auto* out_edge : node->out_edges()) {
      tensor_name2expr.emplace(out_edge->argument().name(), expr);
    }
  };

  tvm::Array<tvm::relay::Expr> fields;
  for (const auto& para : return_params) {
    auto it = tensor_name2expr.find(para.name());
    CHECK(it != tensor_name2expr.end());
    fields.push_back(it->second);
  }
  tvm::NodePtr<tvm::relay::TupleNode> n = tvm::make_node<tvm::relay::TupleNode>();
  n->fields = std::move(fields);
  auto output = tvm::relay::Tuple(n);
  auto graph_func = tvm::relay::FunctionNode::make(input_vars, output, 
      tvm::relay::Type(), {});

  TVMContext ctx;
  ctx.device_type = DLDeviceType::kDLGPU; // only support CUDA GPU now
  ctx.device_id = 0; // TODO(niuchong): how to get the device id?
  auto build_fn = builder_.GetFunction("build", false);
  auto json_fn = builder_.GetFunction("get_graph_json", false);
  auto get_mod_fn = builder_.GetFunction("get_module", false);
  tvm::Map<tvm::Integer, tvm::Target> target_map = {
    {ctx_.device_type, tvm::Target::Create("cuda")}; // only support target as cuda
  };
  build_fn(graph_func, target_map, tvm::Target::Create("llvm"));
  tvm::runtime::Module built_mod = get_mod_fn();
  std::string graph_json = json_fn();

  return std::make_shared<TVMExecutable>(name_, entry_params.size(), return_params,
      graph_json, built_mod, ctx);
}

REGISTER_GRAPH_COMPILER(XrtEngine::TVM, TVMGraphCompiler);

}
}
