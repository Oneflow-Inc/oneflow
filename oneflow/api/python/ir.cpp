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

#ifdef WITH_MLIR
#include "oneflow/ir/include/OneFlow/Extension.h"
#endif  // WITH_MLIR
#include <glog/logging.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include <regex>

namespace {
using namespace oneflow;
using K = std::string;
using V = user_op::OpRegistryResult;

// from llvm
std::string convertToCamelFromSnakeCase(const std::string& input, bool capitalizeFirst) {
  if (input.empty()) return "";

  std::string output;
  output.reserve(input.size());

  // Push the first character, capatilizing if necessary.
  if (capitalizeFirst && std::islower(input.front()))
    output.push_back(toupper(input.front()));
  else
    output.push_back(input.front());

  // Walk the input converting any `*_[a-z]` snake case into `*[A-Z]` camelCase.
  for (size_t pos = 1, e = input.size(); pos < e; ++pos) {
    if (input[pos] == '_' && pos != (e - 1) && std::islower(input[pos + 1]))
      output.push_back(toupper(input[++pos]));
    else
      output.push_back(input[pos]);
  }
  return output;
}

std::string GetMLIRAttrType(const AttrType& attr_type) {
  if (attr_type == ::oneflow::kAtInt32) {
    return "SI32Attr";
  } else if (attr_type == ::oneflow::kAtInt64) {
    return "SI64Attr";
  } else if (attr_type == ::oneflow::kAtBool) {
    return "BoolAttr";
  } else if (attr_type == ::oneflow::kAtFloat) {
    return "F32Attr";
  } else if (attr_type == ::oneflow::kAtDouble) {
    return "F64Attr";
  } else if (attr_type == ::oneflow::kAtString) {
    return "StrAttr";
  } else if (attr_type == ::oneflow::kAtShape) {
    return "AnyI64ElementsAttr";
  } else if (attr_type == ::oneflow::kAtDataType) {
    return "StrAttr";
  } else if (attr_type == ::oneflow::kAtListInt32) {
    return "SI32ArrayAttr";
  } else if (attr_type == ::oneflow::kAtListInt64) {
    return "SI64ArrayAttr";
  } else if (attr_type == ::oneflow::kAtListFloat) {
    return "F32ArrayAttr";
  } else if (attr_type == ::oneflow::kAtListDataType) {
    return "DTArrayAttr";
  } else if (attr_type == ::oneflow::kAtListShape) {
    return "ShapeArrayAttr";
  } else if (attr_type == ::oneflow::kAtListString) {
    return "StrArrayAttr";
  } else {
    LOG(FATAL) << "fail to convert: " << attr_type;
    return "failure";
  }
}

const std::set<std::string>& GetIdempotentOps() {
  static std::set<std::string> ret{"abs",   "ceil", "floor", "ones_like", "relu",      "relu_grad",
                                   "relu6", "rint", "round", "sign",      "zeros_like"};
  return ret;
}
const std::set<std::string>& GetInvolutionOps() {
  static std::set<std::string> ret{"reciprocal", "negative"};
  return ret;
}

bool IsGradOp(const std::string& op_name) { return op_name.find("grad") != std::string::npos; }

const std::set<std::string>& GetMathOps() {
  static std::set<std::string> ret{"abs",         "acos",
                                   "acosh",       "asin",
                                   "asinh",       "atan",
                                   "atanh",       "ceil",
                                   "cos",         "cosh",
                                   "erf",         "erfc",
                                   "exp",         "expm1",
                                   "floor",       "lgamma",
                                   "log",         "log1p",
                                   "log_sigmoid", "negative",
                                   "reciprocal",  "reciprocal_no_nan",
                                   "rint",        "round",
                                   "rsqrt",       "sigmoid_v2",
                                   "sign",        "sin",
                                   "sinh",        "softplus",
                                   "sqrt",        "square",
                                   "tan",         "tanh"};
  return ret;
}
bool IsMathOp(const std::string& op_name) {
  bool is_grad = false;
  for (const auto& name : GetMathOps()) {
    if (op_name.find(name) != std::string::npos && IsGradOp(op_name)) { is_grad = true; }
  }
  return GetMathOps().find(op_name) != GetMathOps().end() || is_grad;
}
bool IsInvolutionOp(const std::string& op_name) {
  return GetInvolutionOps().find(op_name) != GetInvolutionOps().end() && !IsGradOp(op_name);
}
bool IsIdempotentOp(const std::string& op_name) {
  return GetIdempotentOps().find(op_name) != GetIdempotentOps().end() && !IsGradOp(op_name);
}

bool IsPoolOp(const std::string& op_name) {
  return (op_name.rfind("avg", 0) == 0 || op_name.rfind("max", 0) == 0)
         && op_name.find("pool") != std::string::npos;
}
bool IsEagerOp(const std::string& op_name) { return (op_name.rfind("eager", 0) == 0); }
bool IsAnyPoolOp(const std::string& op_name) { return op_name.find("pool") != std::string::npos; }
bool IsAnyConvOp(const std::string& op_name) { return op_name.find("conv") != std::string::npos; }
bool IsConvOp(const std::string& op_name) {
  return op_name.rfind("conv", 0) == 0 && op_name.find("grad") == std::string::npos;
}

bool IsLazyPoolOp(const std::string& op_name) { return op_name.find("_pool") != std::string::npos; }
bool IsNCCLOp(const std::string& op_name) { return op_name.find("nccl") != std::string::npos; }
bool IsOptimizerOp(const std::string& op_name) {
  return (op_name.find("update") != std::string::npos || op_name.find("adam") != std::string::npos)
         && op_name.find("scatter") == std::string::npos;
}
bool IsTrigonometric(const std::string& op_name) {
  return (op_name.find("sin") != std::string::npos || op_name.find("cos") != std::string::npos
          || op_name.find("tan") != std::string::npos)
         && op_name.find("constant") == std::string::npos;
}
bool IsTestOp(const std::string& op_name) {
  return (op_name.find("test") != std::string::npos || op_name.find("Test") != std::string::npos
          || op_name.find("ccrelu") != std::string::npos);
}
bool IsPaddingOp(const std::string& op_name) { return (op_name.find("pad") != std::string::npos); }
bool IsAssignOp(const std::string& op_name) {
  return (op_name.find("assign") != std::string::npos);
}
bool IsCrossEntropyOp(const std::string& op_name) {
  return (op_name.find("cross_entropy") != std::string::npos);
}
bool IsMatmulOp(const std::string& op_name) {
  return (op_name.find("matmul") != std::string::npos || op_name.find("fc") != std::string::npos);
}

bool IsDatasetOp(const std::string& op_name) {
  return (op_name.find("reader") != std::string::npos || op_name.find("Reader") != std::string::npos
          || op_name.find("loader") != std::string::npos
          || op_name.find("decoder") != std::string::npos);
}
bool IsUpsampleOp(const std::string& op_name) {
  return (op_name.find("upsample") != std::string::npos);
}
bool IsBroadcastOp(const std::string& op_name) {
  return (op_name.find("broadcast") != std::string::npos);
}
bool IsIdentityOp(const std::string& op_name) {
  return (op_name.find("identity") != std::string::npos);
}
bool IsScalarOp(const std::string& op_name) {
  return (op_name.rfind("scalar_", 0) == 0 || op_name.find("by_scalar") != std::string::npos);
}
bool IsImageOp(const std::string& op_name) { return (op_name.find("image") != std::string::npos); }
bool IsSoftmaxOp(const std::string& op_name) {
  return (op_name.find("softmax") != std::string::npos);
}
bool IsFusedOp(const std::string& op_name) {
  return (op_name.find("fused") != std::string::npos
          || op_name.find("add_relu") != std::string::npos);
}
bool IsReduceOp(const std::string& op_name) {
  return (op_name.find("reduce") != std::string::npos);
}
bool IsReshapeOp(const std::string& op_name) {
  return (op_name.find("reshape") != std::string::npos);
}
bool IsLossOp(const std::string& op_name) { return (op_name.find("loss") != std::string::npos); }
bool IsIndicesOp(const std::string& op_name) {
  return (op_name.find("arg") != std::string::npos || op_name.find("where") != std::string::npos
          || op_name.find("gather") != std::string::npos
          || op_name.find("slice") != std::string::npos
          || op_name.find("segment_sum") != std::string::npos
          || op_name.find("top_k") != std::string::npos
          || op_name.find("scatter") != std::string::npos);
}
bool IsNormalizationOp(const std::string& op_name) {
  return (op_name.find("norm") != std::string::npos);
}

std::string PostProcessClassName(const std::string& op_name) {
  std::string ret = op_name;
  ret = std::regex_replace(ret, std::regex("pool"), "Pool");
  ret = std::regex_replace(ret, std::regex("_1d"), "1D");
  ret = std::regex_replace(ret, std::regex("_2d"), "2D");
  ret = std::regex_replace(ret, std::regex("_3d"), "3D");
  ret = std::regex_replace(ret, std::regex("1d"), "1D");
  ret = std::regex_replace(ret, std::regex("2d"), "2D");
  ret = std::regex_replace(ret, std::regex("3d"), "3D");
  return ret;
}

std::string GetPoolOpClassName(const std::string& op_name) {
  std::string ret((IsLazyPoolOp(op_name) ? "Lazy" : "Eager")
                  + convertToCamelFromSnakeCase(op_name, true));
  return ret;
}

std::string GetConvOpClassName(const std::string& op_name) {
  std::string ret(convertToCamelFromSnakeCase(op_name, true));
  // NOTE: should change form conv => Convolution ?
  return ret;
}

std::string GetBaseOp(const std::string& op_name) {
  if (IsInvolutionOp(op_name)) {
    return "OneFlow_InvolutionBaseOp";
  } else if (IsIdempotentOp(op_name)) {
    return "OneFlow_IdempotentBaseOp";
  } else if (IsConvOp(op_name)) {
    return "OneFlow_ConvolutionBaseOp";
  } else if (IsPoolOp(op_name)) {
    return "OneFlow_" + std::string(IsLazyPoolOp(op_name) ? "Lazy" : "Eager") + "Pool"
           + std::string(IsGradOp(op_name) ? "Grad" : "") + "BaseOp";
  } else {
    return "OneFlow_BaseOp";
  }
}

bool ShouldGenEmptyBody(const std::string& op_name) {
  return false;
  // return IsInvolutionOp(op_name) || IsIdempotentOp(op_name) || IsPoolOp(op_name)
  //        || IsConvOp(op_name);
}

void PrintArgDef(const UserOpDef_ArgDef& arg_def) {
  std::cout << "    ";
  if (arg_def.is_optional()) { std::cout << "Optional<"; }
  if (arg_def.num_as_min()) { std::cout << "Variadic<"; }
  std::cout << "AnyType";
  if (arg_def.is_optional() || arg_def.num_as_min()) { std::cout << ">"; }
  CHECK(!(arg_def.is_optional() && arg_def.num_as_min())) << arg_def.DebugString();
  std::cout << ":$" << arg_def.name();
  if (arg_def.num_as_min()) {
    // TODO: add verifier
  }
}

bool HasMultipleVariadic(
    const ::google::protobuf::RepeatedPtrField<::oneflow::UserOpDef_ArgDef>& arg_defs) {
  uint32_t num_variadic_op = 0;
  for (const auto& arg_def : arg_defs) {
    if (arg_def.is_optional()) { num_variadic_op += 1; }
    if (arg_def.num_as_min()) { num_variadic_op += 1; }
  }
  return num_variadic_op > 1;
}

std::string GetOperandOrder(
    const ::google::protobuf::RepeatedPtrField<::oneflow::UserOpDef_ArgDef>& arg_defs) {
  std::string ret = "{";
  for (auto it = arg_defs.begin(); it != arg_defs.end(); ++it) {
    ret += ("\"" + it->name() + "\"");
    if (std::next(it) != arg_defs.end()) { ret += ", "; }
  }
  ret += "}";
  return ret;
}

// TODO: use MLIR Interfaces it implement this
void PrintExtraClassDeclaration(const oneflow::UserOpDef& op_def) {
  std::cout << "  let extraClassDeclaration = [{"
            << "\n";
  std::cout << "    static std::vector<std::string> inputOrder() { return "
            << GetOperandOrder(op_def.input()) << "; }\n";
  std::cout << "    static std::vector<std::string> outputOrder() { return "
            << GetOperandOrder(op_def.output()) << "; }\n";
  std::cout << "  }];"
            << "\n";
}

void PrintHasCanonicalizer(const std::string& op_name) {
  if (op_name == "add_n") {
    std::cout << "  let hasCanonicalizer = 1;"
              << "\n";
  }
}

void PrintTraitAttrs(const oneflow::UserOpDef& op_def) {
  const bool need_operand_segment_sizes = HasMultipleVariadic(op_def.input());
  const bool need_result_segment_sizes = HasMultipleVariadic(op_def.output());
  if (need_operand_segment_sizes || need_result_segment_sizes) {
    std::cout << "  let trait_attrs = (ins"
              << "\n";
    if (need_operand_segment_sizes) {
      std::cout << "    I32ElementsAttr:$operand_segment_sizes"
                << (need_result_segment_sizes ? ",\n" : "\n");
    }
    if (need_result_segment_sizes) { std::cout << "    I32ElementsAttr:$result_segment_sizes\n"; }
    std::cout << "  );"
              << "\n";
  }
}

void PrintBody(const oneflow::user_op::OpRegistryResult& r) {
  const oneflow::UserOpDef& op_def = r.op_def;
  // TODO: handle in out size/optional
  // TODO: handle "," in last element
  std::cout << "{"
            << "\n";
  // inputs
  if (op_def.input().size()) {
    std::cout << "  let input = (ins"
              << "\n";
    for (auto it = op_def.input().begin(); it != op_def.input().end(); ++it) {
      PrintArgDef(*it);
      std::cout << (std::next(it) == op_def.input().end() ? "" : ",") << "\n";
    }
    std::cout << "  );"
              << "\n";
  }
  // outputs
  if (op_def.output().size()) {
    std::cout << "  let output = (outs"
              << "\n";
    for (auto it = op_def.output().begin(); it != op_def.output().end(); ++it) {
      PrintArgDef(*it);
      std::cout << (std::next(it) == op_def.output().end() ? "" : ",") << "\n";
    }
    std::cout << "  );"
              << "\n";
  }
  // attrs
  if (op_def.attr().size()) {
    std::cout << "  let attrs = (ins"
              << "\n";
    for (auto it = op_def.attr().begin(); it != op_def.attr().end(); ++it) {
      std::cout << "    " << GetMLIRAttrType(it->type()) << ":$" << it->name()
                << (std::next(it) == op_def.attr().end() ? "" : ",") << "\n";
    }
    std::cout << "  );"
              << "\n";
  }
  // trait attrs
  PrintTraitAttrs(op_def);
  PrintExtraClassDeclaration(op_def);
  PrintHasCanonicalizer(r.op_type_name);
  std::cout << "}"
            << "\n";
}

bool ShouldGenBaseClass(const std::string& op_name) { return op_name == "normalization_add_relu"; }

std::string GetOpClassName(const std::string& op_name) {
  std::string ret = "";
  if (IsPoolOp(op_name)) {
    ret = GetPoolOpClassName(op_name);
  } else if (IsConvOp(op_name)) {
    ret = GetConvOpClassName(op_name);
  } else {
    ret = convertToCamelFromSnakeCase(op_name, true);
  }
  if (ShouldGenBaseClass(op_name)) { ret += "Base"; }
  return PostProcessClassName(ret);
}

std::string GetTraits(const oneflow::UserOpDef& op_def) {
  std::string ret{};
  const bool need_operand_segment_sizes = HasMultipleVariadic(op_def.input());
  const bool need_result_segment_sizes = HasMultipleVariadic(op_def.output());
  if (need_operand_segment_sizes) { ret += "AttrSizedOperandSegments"; }

  if (need_result_segment_sizes) {
    if (ret != "") ret += ", ";
    ret += "AttrSizedResultSegments";
  }
  if (ret != "") ret += ", ";
  ret += "DeclareOpInterfaceMethods<BnOrderOpInterface>";
  return ret;
}

bool IsReferencedByOtherDefinitions(const std::string& op_name) {
  return ShouldGenBaseClass(op_name);
}

void PrintODSFromOpRegistryResults(const std::map<K, V>& results) {
  for (const auto& kv : results) {
    if (kv.first == "mlir_jit") continue;
    const oneflow::user_op::OpRegistryResult& r = kv.second;
    auto op_class_name = GetOpClassName(kv.first);
    std::cout << (ShouldGenBaseClass(r.op_type_name) ? "class" : "def") << " OneFlow_"
              << op_class_name << "Op : " << GetBaseOp(r.op_type_name) << "<\"" << kv.first
              << "\", [" + GetTraits(r.op_def) + "]> ";  // TODO: add traits
    if (ShouldGenEmptyBody(r.op_type_name)) {
      std::cout << "{}\n";
    } else {
      PrintBody(r);
    }
    std::cout << "\n";
  }
}

void GroupOpRegistryResults(const std::map<K, V>& results,
                            std::map<std::string, std::map<K, V>>& groups) {
  for (const auto& kv : results) {
    std::string group_name = "MISC";
    const oneflow::user_op::OpRegistryResult& r = kv.second;
    if (ShouldGenBaseClass(r.op_type_name)) { group_name = "BASE"; }
    if (IsImageOp(r.op_type_name)) { group_name = "Image"; }
    if (IsMathOp(r.op_type_name)) { group_name = "math"; }
    if (IsPaddingOp(r.op_type_name)) { group_name = "PADDING"; }
    if (IsIndicesOp(r.op_type_name)) { group_name = "Indices"; }
    if (IsBroadcastOp(r.op_type_name)) { group_name = "Broadcast"; }
    if (IsScalarOp(r.op_type_name)) { group_name = "Scalar"; }
    if (IsReduceOp(r.op_type_name)) { group_name = "reduce"; }
    if (IsReshapeOp(r.op_type_name)) { group_name = "reshape"; }
    if (IsLossOp(r.op_type_name)) { group_name = "loss"; }
    if (IsNormalizationOp(r.op_type_name)) { group_name = "Normalization"; }
    if (IsCrossEntropyOp(r.op_type_name)) { group_name = "Cross_Entropy"; }
    if (IsSoftmaxOp(r.op_type_name)) { group_name = "Softmax"; }
    if (IsNCCLOp(r.op_type_name)) { group_name = "NCCL"; }
    if (IsAnyConvOp(r.op_type_name)) { group_name = "CONV"; }
    if (IsAnyPoolOp(r.op_type_name)) { group_name = "POOL"; }
    if (IsUpsampleOp(r.op_type_name)) { group_name = "UPSAMPLE"; }
    if (IsAssignOp(r.op_type_name)) { group_name = "assign"; }
    if (IsOptimizerOp(r.op_type_name)) { group_name = "OPTIMIZER"; }
    if (IsTrigonometric(r.op_type_name)) { group_name = "TRIGONOMETRIC"; }
    if (IsIdempotentOp(r.op_type_name)) { group_name = "IDEMPOTENT"; }
    if (IsInvolutionOp(r.op_type_name)) { group_name = "INVOLUTION"; }
    if (IsIdentityOp(r.op_type_name)) { group_name = "Identity"; }
    if (IsFusedOp(r.op_type_name)) { group_name = "Fused"; }
    if (IsEagerOp(r.op_type_name)) { group_name = "eager"; }
    if (IsDatasetOp(r.op_type_name)) { group_name = "DATASET"; }
    if (IsMatmulOp(r.op_type_name)) { group_name = "matmul"; }
    if (IsTestOp(r.op_type_name)) { group_name = "TEST"; }
    group_name = "GET_ONEFLOW_" + group_name + "_OP_DEFINITIONS";
    std::transform(group_name.begin(), group_name.end(), group_name.begin(), ::toupper);
    groups[group_name].insert({kv.first, kv.second});
  }
}

}  // namespace

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("ir", m) {
#ifdef WITH_MLIR
  m.def("load_jit_shared_lib",
        [](const std::string& lib_path) { MutSharedLibPaths()->insert(lib_path); });
#endif  // WITH_MLIR
  m.def("gen_ods", []() {
    using K = std::string;
    using V = user_op::OpRegistryResult;

    std::map<K, V> sorted{};
    auto unordered = user_op::UserOpRegistryMgr::Get().GetAllOpRegistryResults();
    std::transform(unordered.begin(), unordered.end(), std::inserter(sorted, sorted.end()),
                   [](const std::pair<K, V>& p) { return p; });
    std::map<std::string, std::map<K, V>> groups;
    GroupOpRegistryResults(sorted, groups);
    for (const auto& kv : groups) {
      const auto& group_name = kv.first;
      std::cout << "// "
                << "Total: " << kv.second.size() << "\n";
      std::cout << "#ifndef " << group_name << "\n";
      std::cout << "#define " << group_name << "\n\n";
      auto results = kv.second;
      PrintODSFromOpRegistryResults(results);
      std::cout << "#endif  // " << group_name << "\n\n";
    }
  });
}

}  // namespace oneflow
