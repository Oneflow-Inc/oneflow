/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/xrt/tvm/tvm_graph_compiler.h"
#include "oneflow/xrt/tvm/tvm_executable.h"
#include "oneflow/xrt/tvm/ops/op_context.h"
#include "oneflow/xrt/tvm/ops/op_kernel.h"
#include "oneflow/xrt/node_util.h"
#include <tvm/runtime/device_api.h>
#include <tvm/target/target.h>
#include <tvm/node/node.h>
#include <tuple>

namespace oneflow {
namespace xrt {
namespace of_tvm {

namespace {

tvm::Array<tvm::relay::IndexExpr> ConvertShapeToTVM(const oneflow::Shape& shape) {
  tvm::Array<tvm::relay::IndexExpr> ret;
  for (int i = 0; i < shape.NumAxes(); ++i) {
    ret.push_back(tvm::relay::IndexExpr(static_cast<int32_t>(shape.At(i))));
  }
  return ret;
}

tvm::runtime::DataType ConvertDataTypeToTVM(DataType dtype) {
  using tvmDataType = tvm::runtime::DataType;
  switch (dtype) {
    case DataType::kChar: return tvmDataType(tvmDataType::kUInt, 8, 1); break;
    case DataType::kFloat: return tvmDataType(tvmDataType::kFloat, 32, 1); break;
    case DataType::kInt8: return tvmDataType(tvmDataType::kInt, 8, 1); break;
    case DataType::kInt32: return tvmDataType(tvmDataType::kInt, 32, 1); break;
    case DataType::kInt64: return tvmDataType(tvmDataType::kInt, 64, 1); break;
    case DataType::kUInt8: return tvmDataType(tvmDataType::kUInt, 8, 1); break;
    case DataType::kFloat16: return tvmDataType(tvmDataType::kFloat, 16, 1); break;
    default: LOG(FATAL) << "Unsupported DataType: " << dtype;
  }
}
// DLDataType

void ConvertEntryParamsToTVMExpr(const std::vector<Parameter>& entry_params,
                                 util::Map<std::string, tvm::relay::Expr>* tensor_name2expr,
                                 tvm::Array<tvm::relay::Var>* graph_input_vars) {
  for (const auto& para : entry_params) {
    auto tensor_type = tvm::relay::TensorType(ConvertShapeToTVM(para.shape()),
                                              ConvertDataTypeToTVM(para.data_type()));
    auto var = tvm::relay::Var(para.name(), tensor_type);
    CHECK(tensor_name2expr->emplace(para.name(), var).second);
    graph_input_vars->push_back(var);
  }
}

tvm::relay::Expr ConvertReturnParamsToTVMExpr(
    const std::vector<Parameter>& return_params,
    const util::Map<std::string, tvm::relay::Expr>& tensor_name2expr) {
  tvm::Array<tvm::relay::Expr> fields;
  for (const auto& para : return_params) {
    auto it = tensor_name2expr.find(para.name());
    CHECK(it != tensor_name2expr.end());
    fields.push_back(it->second);
  }
  return tvm::relay::Tuple(fields);
}

std::tuple<tvm::runtime::Module, std::string> BuildGraphModule(tvm::relay::Function graph_func,
                                                               const XrtDevice& device) {
  auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
  tvm::runtime::Module build_mod = (*pfb)();
  auto build_f = build_mod.GetFunction("build", false);
  auto json_f = build_mod.GetFunction("get_graph_json", false);
  auto mod_f = build_mod.GetFunction("get_module", false);

  tvm::Map<tvm::Integer, tvm::Target> targets;
  if (device == XrtDevice::GPU_CUDA) {
    targets.Set(DLDeviceType::kDLGPU, tvm::Target("cuda"));
  } else if (device == XrtDevice::CPU_X86) {
    targets.Set(DLDeviceType::kDLCPU, tvm::Target("llvm"));
  } else {
    LOG(FATAL) << "Unsupported XrtDevice: " << device;
  }

  // use IRModule instead of function since no parameters are specified
  auto relay_mod = tvm::IRModule::FromExpr(graph_func);
  build_f(relay_mod, targets, tvm::Target("llvm"));

  std::string json = json_f();
  tvm::runtime::Module mod = mod_f();
  return std::make_tuple(std::move(mod), std::move(json));
}

}  // namespace

TVMGraphCompiler::TVMGraphCompiler(const std::string& name) : GraphCompiler::Impl(name) {}

std::shared_ptr<Executable> TVMGraphCompiler::Compile(
    const XrtGraph* graph, const std::vector<Parameter>& entry_params,
    const std::vector<Parameter>& return_params, const std::vector<InputOutputAlias>& aliases) {
  util::Map<std::string, tvm::relay::Expr> tensor_name2expr;
  tvm::Array<tvm::relay::Var> graph_input_vars;
  LOG(WARNING) << "Compile Xrt graph with TVM";
  // VLOG(3) << graph->ToDot();

  ConvertEntryParamsToTVMExpr(entry_params, &tensor_name2expr, &graph_input_vars);

  algorithm::TopologyVisit(*graph, [&](const XrtNode* node) {
    if (node->IsArgumentNode()) { return; }

    VLOG(3) << "TVM compiling node <" << node->type() << ">:" << node->name();
    util::Map<Argument, tvm::relay::Expr> input_arg2expr;
    for (const auto* in_edge : node->in_edges()) {
      const Argument& in_arg = in_edge->argument();
      auto it = tensor_name2expr.find(in_arg.name());
      CHECK(it != tensor_name2expr.end());
      input_arg2expr.emplace(in_arg, it->second);
    }
    util::Vector<Argument> out_args;
    for (const auto* out_edge : node->out_edges()) {
      const auto& out_arg = out_edge->argument();
      out_args.push_back(out_arg);
    }

    TVMOpContext ctx(node, OpMessage(node), std::move(input_arg2expr), std::move(out_args));
    auto op_kernel = BuildTVMOpKernel(this->device_, node->type());
    op_kernel->Compile(&ctx);

    for (const auto* out_edge : node->out_edges()) {
      const auto& out_arg = out_edge->argument();
      if (tensor_name2expr.find(out_arg.name()) == tensor_name2expr.end()) {
        const std::string& produce_key = out_arg.meta_data().produce_key;
        tvm::relay::Expr op_expr = ctx.GetExpr4OutputName(produce_key);
        CHECK(op_expr.defined()) << "Get an empty tvm expression for output: " << produce_key
                                 << " of node: " << node->name();
        CHECK(tensor_name2expr.emplace(out_arg.name(), op_expr).second);
      }
    }
  });

  auto outputs = ConvertReturnParamsToTVMExpr(return_params, tensor_name2expr);
  auto graph_func = tvm::relay::Function(graph_input_vars, outputs, tvm::relay::Type(), {});

  tvm::runtime::Module built_mod;
  std::string graph_json;

  std::tie(built_mod, graph_json) = BuildGraphModule(graph_func, this->device_);
  // VLOG(3) << "Got TVM graph_json:\n" << graph_json;

  return std::make_shared<TVMExecutable>(this->name_, entry_params.size(), return_params,
                                         graph_json, built_mod, this->device_);
}

REGISTER_GRAPH_COMPILER(XrtEngine::TVM, TVMGraphCompiler);

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow