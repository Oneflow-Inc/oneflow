#include "oneflow/xrt/tvm/tvm_graph_compiler.h"
#include "oneflow/xrt/tvm/tvm_executable.h"
#include <tvm/runtime/device_api.h>
#include <tuple>

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
  switch (dtype) {
    case DataType::kChar: return tvm::Int(8); break;
    case DataType::kFloat: return tvm::Float(32); break;
    case DataType::kDouble: return tvm::Float(64); break;
    case DataType::kInt8: return tvm::Int(8); break;
    case DataType::kInt32: return tvm::Int(32); break;
    case DataType::kInt64: return tvm::Int(64); break;
    case DataType::kUInt8: return tvm::UInt(8); break;
    case DataType::kFloat16: return tvm::Float(16); break;
    default: LOG(FATAL) << "Unsupported DataType: " << dtype;
  }
}

void ConvertEntryParamsToTVMExpr(const std::vector<Parameter>& entry_params,
    util::Map<std::string, tvm::relay::Expr>* tensor_name2expr,
    tvm::Array<tvm::relay::Var>* graph_input_vars) {
  for (const auto& para : entry_params) {
    auto tensor_type = tvm::relay::TensorTypeNode::make(
        ConvertShapeToTVM(para.shape()),
        ConvertDataTypeToTVM(para.data_type()));
    auto var = tvm::relay::VarNode::make(para.name(),
        tensor_type);
    tensor_name2expr->emplace(para.name(), var);
    graph_input_vars->push_back(var);
  }
}

tvm::relay::Expr ConvertReturnParamsToTVMExpr(const std::vector<Parameter>& return_params,
    const util::Map<std::string, tvm::relay::Expr>& tensor_name2expr) {
  tvm::Array<tvm::relay::Expr> fields;
  for (const auto& para : return_params) {
    auto it = tensor_name2expr.find(para.name());
    CHECK(it != tensor_name2expr.end());
    fields.push_back(it->second);
  }
  tvm::NodePtr<tvm::relay::TupleNode> n = tvm::make_node<tvm::relay::TupleNode>();
  n->fields = std::move(fields);
  return tvm::relay::Tuple(n);
}

tvm::relay::Expr ConvertOpNodeToTVMExpr(const XrtNode* node,
    const tvm::Array<tvm::relay::Expr>& node_inputs) {
}

std::tuple<tvm::runtime::Module, std::string>
    BuildGraphModule(tvm::relay::Function graph_function) {
  auto create_fn = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
  tvm::runtime::Module builder = (*create_fn)();
  auto build_fn = builder.GetFunction("build", false);
  auto json_fn = builder.GetFunction("get_graph_json", false);
  auto get_mod_fn = builder.GetFunction("get_module", false);

  tvm::Map<tvm::Integer, tvm::Target> target_map = {
    {DLDeviceType::kDLGPU, tvm::Target::Create("cuda")}}; //TODO(niuchong): support more devs and targets
  build_fn(graph_function, target_map, tvm::Target::Create("llvm"));
  tvm::runtime::Module built_mod = get_mod_fn();
  std::string graph_json = json_fn();
  return std::make_tuple(std::move(built_mod), std::move(graph_json));
}

}

TVMGraphCompiler::TVMGraphCompiler(const std::string& name) : GraphCompiler::Impl(name) {}

std::shared_ptr<Executable> TVMGraphCompiler::Compile(const XrtGraph *graph,
                                    const std::vector<Parameter> &entry_params,
                                    const std::vector<Parameter> &return_params,
                                    const std::vector<InputOutputAlias> &aliases) {
  util::Map<std::string, tvm::relay::Expr> tensor_name2expr;
  tvm::Array<tvm::relay::Var> graph_input_vars;

  ConvertEntryParamsToTVMExpr(entry_params, &tensor_name2expr, &graph_input_vars);

  algorithm::TopologyVisit(*graph, [&](const XrtNode* node) {
    if (node->IsArgumentNode()) { return; }
    tvm::Array<tvm::relay::Expr> node_inputs;
    for (const auto* in_edge : node->in_edges()) {
      auto it = tensor_name2expr.find(in_edge->argument().name());
      CHECK(it != tensor_name2expr.end());
      node_inputs.push_back(it->second);
    }
    auto expr = ConvertOpNodeToTVMExpr(node, node_inputs);
    for (const auto* out_edge : node->out_edges()) {
      tensor_name2expr.emplace(out_edge->argument().name(), expr);
    }
  });

  auto outputs = ConvertReturnParamsToTVMExpr(return_params, tensor_name2expr);
  auto graph_func = tvm::relay::FunctionNode::make(graph_input_vars, outputs, 
      tvm::relay::Type(), {});

  tvm::runtime::Module built_mod;
  std::string graph_json;
  std::tie(built_mod, graph_json) = BuildGraphModule(graph_func);

  return std::make_shared<TVMExecutable>(this->name_, entry_params.size(), return_params,
      graph_json, built_mod, XrtDevice::GPU_CUDA); // only support GPU_CUDA now
}

REGISTER_GRAPH_COMPILER(XrtEngine::TVM, TVMGraphCompiler);

}
}
