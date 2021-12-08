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

#include <glog/logging.h>
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include <regex>

namespace {

using K = std::string;
using V = ::oneflow::user_op::OpRegistryResult;
using ::oneflow::AttrType;
using ::oneflow::UserOpDef_ArgDef;

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

std::string GetMLIRAttrTypeName(const AttrType& attr_type) {
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
    return "OneFlow_DataType";
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

template<typename T>
std::string ToZeroNoTrailing(T f) {
  std::string str = std::to_string(f);
  str.erase(str.find_last_not_of('0') + 1, std::string::npos);
  return str;
}

std::string GetDefaultValue(const ::oneflow::AttrValue& attr_val) {
  if (attr_val.has_at_string()) {
    return "\\\"" + attr_val.at_string() + "\\\"";
  } else if (attr_val.has_at_int32()) {
    return std::to_string(attr_val.at_int32());
  } else if (attr_val.has_at_int64()) {
    return std::to_string(attr_val.at_int64());
  } else if (attr_val.has_at_float()) {
    return ToZeroNoTrailing(attr_val.at_float());
  } else if (attr_val.has_at_double()) {
    return ToZeroNoTrailing(attr_val.at_double());
  } else if (attr_val.has_at_bool()) {
    return attr_val.at_bool() ? "true" : "false";
  } else if (attr_val.has_at_list_int32()) {
    std::string ret = "{";
    const auto& list = attr_val.at_list_int32().val();
    for (auto it = list.begin(); it != list.end(); ++it) {
      ret += std::to_string(*it) + (std::next(it) == list.end() ? "" : ", ");
    }
    ret += "}";
    return ret;
  } else if (attr_val.has_at_list_int64()) {
    std::string ret = "{";
    const auto& list = attr_val.at_list_int64().val();
    for (auto it = list.begin(); it != list.end(); ++it) {
      ret += std::to_string(*it) + (std::next(it) == list.end() ? "" : ", ");
    }
    ret += "}";
    return ret;
  } else if (attr_val.has_at_list_float()) {
    std::string ret = "{";
    const auto& list = attr_val.at_list_float().val();
    for (auto it = list.begin(); it != list.end(); ++it) {
      ret += std::to_string(*it) + (std::next(it) == list.end() ? "" : ", ");
    }
    ret += "}";
    return ret;
  } else if (attr_val.has_at_list_string()) {
    std::string ret = "{";
    const auto& list = attr_val.at_list_string().val();
    for (auto it = list.begin(); it != list.end(); ++it) {
      ret += "\"" + *it + "\"" + (std::next(it) == list.end() ? "" : ", ");
    }
    ret += "}";
    return ret;
  } else if (attr_val.has_at_data_type()) {
    return std::to_string(attr_val.at_data_type());
  }
  LOG(FATAL) << "fail to convert value_case: " << attr_val.value_case() << "\n"
             << attr_val.DebugString();
}

std::string GetMLIRAttrType(const ::oneflow::UserOpDef_AttrDef& attr_def) {
  const AttrType& attr_type = attr_def.type();
  std::string name = GetMLIRAttrTypeName(attr_type);
  auto is_default_supported =
      attr_def.default_val().has_at_bool() || attr_def.default_val().has_at_int32()
      || attr_def.default_val().has_at_int64() || attr_def.default_val().has_at_float()
      || attr_def.default_val().has_at_double()
      || (attr_def.default_val().has_at_string() && attr_def.default_val().at_string().size() > 0);
  if (attr_def.has_default_val() && is_default_supported) {
    name =
        "DefaultValuedAttr<" + name + ", " + "\"" + GetDefaultValue(attr_def.default_val()) + "\">";
  }
  return name;
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
const std::set<std::string>& GetQuantizationOps() {
  static std::set<std::string> ret{"min_max_observer", "moving_average_min_max_observer",
                                   "fake_quantization", "quantization"};
  return ret;
}

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

const std::set<std::string>& GetOpsUsedInPatterns() {
  static std::set<std::string> ret{"scalar_mul_by_tensor", "cast",    "tril",    "scalar_mul",
                                   "fused_scale_tril",     "dropout", "bias_add"};
  return ret;
}
bool IsMathOp(const std::string& op_name) {
  bool is_grad = false;
  for (const auto& name : GetMathOps()) {
    if (op_name.find(name) != std::string::npos && IsGradOp(op_name)) { is_grad = true; }
  }
  return GetMathOps().find(op_name) != GetMathOps().end() || is_grad;
}
bool IsUsedInPatterns(const std::string& op_name) {
  return GetOpsUsedInPatterns().find(op_name) != GetOpsUsedInPatterns().end();
}
bool IsInvolutionOp(const std::string& op_name) {
  return GetInvolutionOps().find(op_name) != GetInvolutionOps().end() && !IsGradOp(op_name);
}
bool IsQuantizationOp(const std::string& op_name) {
  return GetQuantizationOps().find(op_name) != GetQuantizationOps().end();
}
bool IsIdempotentOp(const std::string& op_name) {
  return GetIdempotentOps().find(op_name) != GetIdempotentOps().end() && !IsGradOp(op_name);
}

bool IsPoolOp(const std::string& op_name) {
  return ((op_name.rfind("avg", 0) == 0 || op_name.rfind("max", 0) == 0)
          || ((op_name.find("avg") != std::string::npos || op_name.find("max") != std::string::npos)
              && op_name.rfind("tf", 0) == 0))
         && op_name.find("pool") != std::string::npos;
}
bool IsEagerOp(const std::string& op_name) { return (op_name.rfind("eager", 0) == 0); }
bool IsTensorBufferOp(const std::string& op_name) {
  return op_name.find("tensor_buffer") != std::string::npos;
}
bool IsSummaryOp(const std::string& op_name) {
  return op_name.find("summary") != std::string::npos;
}
bool IsAnyPoolOp(const std::string& op_name) { return op_name.find("pool") != std::string::npos; }
bool IsAnyConvOp(const std::string& op_name) { return op_name.find("conv") != std::string::npos; }
bool IsConvOp(const std::string& op_name) {
  return op_name.rfind("conv", 0) == 0 && op_name.find("grad") == std::string::npos;
}

bool IsLazyPoolOp(const std::string& op_name) {
  return op_name.find("_pool") != std::string::npos && op_name.find("tf_") != std::string::npos;
}
bool IsAdaptivePoolOp(const std::string& op_name) {
  return op_name.find("_pool") != std::string::npos
         && op_name.find("adaptive_") != std::string::npos;
}
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
bool IsCUDAOp(const std::string& op_name) { return (op_name.find("nvtx") != std::string::npos); }
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
bool IsDetectionOp(const std::string& op_name) {
  return (op_name.find("top_k") != std::string::npos || op_name.find("bbox") != std::string::npos
          || op_name.find("segmentation") != std::string::npos
          || op_name.find("roi") != std::string::npos || op_name.find("poly") != std::string::npos
          || op_name.find("nms") != std::string::npos
          || op_name.find("object") != std::string::npos);
}
bool IsIndicesOp(const std::string& op_name) {
  return (op_name.find("arg") != std::string::npos || op_name.find("where") != std::string::npos
          || op_name.find("gather") != std::string::npos
          || op_name.find("slice") != std::string::npos
          || op_name.find("indices") != std::string::npos
          || op_name.find("segment_sum") != std::string::npos
          || op_name.find("scatter") != std::string::npos);
}
bool IsNormalizationOp(const std::string& op_name) {
  return (op_name.find("norm") != std::string::npos);
}
bool IsParallelCastOp(const std::string& op_name) {
  return (op_name.find("parallel_cast") != std::string::npos);
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
    return "OneFlow_" + std::string(IsLazyPoolOp(op_name) ? "TF" : "") + "Pool"
           + std::string(IsGradOp(op_name) ? "Grad" : "") + "BaseOp";
  } else if (IsAdaptivePoolOp(op_name)) {
    return "OneFlow_AdaptivePool" + std::string(IsGradOp(op_name) ? "Grad" : "") + "BaseOp";
  } else {
    return "OneFlow_BaseOp";
  }
}

bool ShouldSkipOperandAndResultsAndAttrs(const std::string& op_name) {
  return IsInvolutionOp(op_name) || IsIdempotentOp(op_name);
}

bool ShouldGenEmptyBody(const std::string& op_name) {
  return IsPoolOp(op_name) || IsAdaptivePoolOp(op_name) || IsConvOp(op_name);
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

uint32_t NumMultipleVariadic(
    const ::google::protobuf::RepeatedPtrField<::oneflow::UserOpDef_ArgDef>& arg_defs) {
  uint32_t num_variadic_op = 0;
  for (const auto& arg_def : arg_defs) {
    if (arg_def.is_optional()) { num_variadic_op += 1; }
    if (arg_def.num_as_min()) { num_variadic_op += 1; }
  }
  return num_variadic_op;
}

bool HasAtLeastTwoVariadic(
    const ::google::protobuf::RepeatedPtrField<::oneflow::UserOpDef_ArgDef>& arg_defs) {
  return NumMultipleVariadic(arg_defs) > 1;
}

bool HasVariadic(
    const ::google::protobuf::RepeatedPtrField<::oneflow::UserOpDef_ArgDef>& arg_defs) {
  return NumMultipleVariadic(arg_defs) > 0;
}

std::string GetOperandKeys(
    const ::google::protobuf::RepeatedPtrField<::oneflow::UserOpDef_ArgDef>& arg_defs) {
  std::string ret = "{";
  for (auto it = arg_defs.begin(); it != arg_defs.end(); ++it) {
    ret += ("\"" + it->name() + "\"");
    if (std::next(it) != arg_defs.end()) { ret += ", "; }
  }
  ret += "}";
  return ret;
}

std::string GetOperandMinimums(
    const ::google::protobuf::RepeatedPtrField<::oneflow::UserOpDef_ArgDef>& arg_defs) {
  std::string ret = "{";
  for (auto it = arg_defs.begin(); it != arg_defs.end(); ++it) {
    uint32_t min = 0;
    if (it->is_optional()) {
      min = 0;
    } else if (it->has_num_as_min()) {
      min = it->num();
    } else {
      min = 1;
    }
    ret += std::to_string(min);
    if (std::next(it) != arg_defs.end()) { ret += ", "; }
  }
  ret += "}";
  return ret;
}

// TODO: use MLIR Interfaces it implement this
void PrintReturnStaticVal(const std::string& type, const std::string& func_name,
                          const std::string& val) {
  std::cout << "    static const " + type + "* " + func_name + "() { static " + type + " val(" + val
                   + "); return &val; }\n";
}
void PrintExtraClassDeclaration(const ::oneflow::UserOpDef& op_def) {
  return;
  std::cout << "  let extraClassDeclaration = [{"
            << "\n";
  PrintReturnStaticVal("std::vector<std::string>", "inputKeys", GetOperandKeys(op_def.input()));
  PrintReturnStaticVal("std::vector<std::uint32_t>", "inputMinimums",
                       GetOperandMinimums(op_def.input()));
  PrintReturnStaticVal("std::vector<std::string>", "outputKeys", GetOperandKeys(op_def.output()));
  PrintReturnStaticVal("std::vector<std::uint32_t>", "outputMinimums",
                       GetOperandMinimums(op_def.input()));
  std::cout << "  }];"
            << "\n";
}

void PrintHasCanonicalizer(const std::string& op_name) {
  if (op_name == "add_n") {
    std::cout << "  let hasCanonicalizer = 1;"
              << "\n";
  }
}

void PrintTraitAttrs(const ::oneflow::UserOpDef& op_def) {
  const bool need_operand_segment_sizes = HasAtLeastTwoVariadic(op_def.input());
  const bool need_result_segment_sizes = HasAtLeastTwoVariadic(op_def.output());
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

bool IsUnaryOp(const ::oneflow::user_op::OpRegistryResult& r) {
  return NumMultipleVariadic(r.op_def.input()) == 0 && NumMultipleVariadic(r.op_def.output()) == 0
         && r.op_def.input().size() == 1 && r.op_def.output().size() == 1;
}

bool IsBinaryOp(const ::oneflow::user_op::OpRegistryResult& r) {
  return NumMultipleVariadic(r.op_def.input()) == 0 && NumMultipleVariadic(r.op_def.output()) == 0
         && r.op_def.input().size() == 2 && r.op_def.output().size() == 1;
}

void PrintBody(const ::oneflow::user_op::OpRegistryResult& r) {
  const ::oneflow::UserOpDef& op_def = r.op_def;
  // TODO: handle in out size/optional
  // TODO: handle "," in last element
  std::cout << "{"
            << "\n";
  // inputs
  const bool should_skip_operand_and_results_and_attrs =
      ShouldSkipOperandAndResultsAndAttrs(r.op_type_name);
  const bool should_skip_operand = should_skip_operand_and_results_and_attrs;
  const bool should_skip_result = should_skip_operand_and_results_and_attrs;
  const bool should_skip_attrs = should_skip_operand_and_results_and_attrs;
  if (op_def.input().size() && !should_skip_operand) {
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
  if (op_def.output().size() && !should_skip_result) {
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
  if (op_def.attr().size() && !should_skip_attrs) {
    std::cout << "  let attrs = (ins"
              << "\n";
    for (auto it = op_def.attr().begin(); it != op_def.attr().end(); ++it) {
      std::cout << "    " << GetMLIRAttrType(*it) << ":$" << it->name()
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

bool HasSideEffect(const std::string& op_name) {
  return IsAssignOp(op_name) || IsOptimizerOp(op_name);
}

std::string GetOpClassName(const std::string& op_name) {
  std::string ret = "";
  if (IsConvOp(op_name)) {
    ret = GetConvOpClassName(op_name);
  } else {
    ret = convertToCamelFromSnakeCase(op_name, true);
  }
  if (ShouldGenBaseClass(op_name)) { ret += "Base"; }
  return PostProcessClassName(ret);
}

std::string GetTraits(const ::oneflow::user_op::OpRegistryResult& r) {
  const ::oneflow::UserOpDef& op_def = r.op_def;
  std::string ret{};
  if (HasSideEffect(r.op_type_name) == false) { ret += "NoSideEffect"; }
  const bool need_operand_segment_sizes = HasAtLeastTwoVariadic(op_def.input());
  const bool need_result_segment_sizes = HasAtLeastTwoVariadic(op_def.output());
  if (need_operand_segment_sizes) {
    if (ret != "") ret += ", ";
    ret += "AttrSizedOperandSegments";
  }

  if (need_result_segment_sizes) {
    if (ret != "") ret += ", ";
    ret += "AttrSizedResultSegments";
  }
  if (ret != "") ret += ", ";
  ret += "DeclareOpInterfaceMethods<UserOpCompatibleInterface>";
  return ret;
}

bool IsReferencedByOtherDefinitions(const std::string& op_name) {
  return ShouldGenBaseClass(op_name);
}

bool ShoudSkipOp(const std::string& op_name) { return op_name == "mlir_jit"; }

void PrintODSFromOpRegistryResults(const std::map<K, V>& results) {
  for (const auto& kv : results) {
    if (ShoudSkipOp(kv.first)) continue;
    const ::oneflow::user_op::OpRegistryResult& r = kv.second;
    auto op_class_name = GetOpClassName(kv.first);
    std::cout << (ShouldGenBaseClass(r.op_type_name) ? "class" : "def") << " OneFlow_"
              << op_class_name << "Op : " << GetBaseOp(r.op_type_name) << "<\"" << kv.first
              << "\", [" + GetTraits(r) + "]> ";  // TODO: add traits
    if (ShouldGenEmptyBody(r.op_type_name)) {
      std::cout << "{}\n";
    } else {
      PrintBody(r);
    }
    std::cout << "\n";
  }
}

void PrintNamesInResults(const std::map<K, V>& results) {
  std::cout << "// ";
  for (auto it = results.begin(); it != results.end(); ++it) {
    std::cout << it->first;
    if (std::next(it) != results.end()) { std::cout << ", "; }
  }
  std::cout << "\n";
}

void PrintGroupNames(std::map<std::string, std::map<K, V>>& groups) {
  std::cout << "// ";
  for (auto it = groups.begin(); it != groups.end(); ++it) {
    if (ShoudSkipOp(it->first)) continue;
    std::cout << it->first;
    if (std::next(it) != groups.end()) { std::cout << ";"; }
  }
  std::cout << "\n\n";
}

void PrintIncludes(std::map<std::string, std::map<K, V>>& groups) {
  std::cout << "/*\n";
  for (auto it = groups.begin(); it != groups.end(); ++it) {
    auto group_name = it->first;
    if (group_name == "BASE") continue;
    if (group_name == "TEST") continue;
    std::transform(group_name.begin(), group_name.end(), group_name.begin(), ::tolower);
    group_name += "_ops";
    std::cout << "#define GET_OP_LIST\n";
    std::cout << "#include \"OneFlow/OneFlow." << group_name << ".cpp.inc\"\n";
    if (std::next(it) != groups.end()) { std::cout << ",\n"; }
  }
  std::cout << "*/\n\n";
}

void GroupOpRegistryResults(const std::map<K, V>& results,
                            std::map<std::string, std::map<K, V>>& groups) {
  for (const auto& kv : results) {
    std::string group_name = "MISC";
    const ::oneflow::user_op::OpRegistryResult& r = kv.second;
    if (IsUnaryOp(r)) { group_name = "Unary"; }
    if (IsBinaryOp(r)) { group_name = "Binary"; }
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
    if (IsQuantizationOp(r.op_type_name)) { group_name = "QUANTIZATION"; }
    if (IsDatasetOp(r.op_type_name)) { group_name = "DATASET"; }
    if (IsMatmulOp(r.op_type_name)) { group_name = "matmul"; }
    if (IsTensorBufferOp(r.op_type_name)) { group_name = "tensor_buffer"; }
    if (IsTestOp(r.op_type_name)) { group_name = "TEST"; }
    if (IsDetectionOp(r.op_type_name)) { group_name = "Detection"; }
    if (IsSummaryOp(r.op_type_name)) { group_name = "summary"; }
    if (IsCUDAOp(r.op_type_name)) { group_name = "cuda"; }
    if (IsParallelCastOp(r.op_type_name)) { group_name = "parallel_cast"; }
    if (ShouldGenBaseClass(r.op_type_name)) { group_name = "BASE"; }
    // if (IsUsedInPatterns(r.op_type_name)) { group_name = "used_in_patterns"; }
    std::transform(group_name.begin(), group_name.end(), group_name.begin(), ::toupper);
    groups[group_name].insert({kv.first, kv.second});
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  std::streambuf* coutBuf = std::cout.rdbuf();
  std::ofstream of("OneFlowUserOpGen.td");
  std::streambuf* fileBuf = of.rdbuf();
  std::cout.rdbuf(fileBuf);

  std::map<K, V> sorted{};
  auto unordered = oneflow::user_op::UserOpRegistryMgr::Get().GetAllOpRegistryResults();
  std::transform(unordered.begin(), unordered.end(), std::inserter(sorted, sorted.end()),
                 [](const std::pair<K, V>& p) { return p; });
  std::map<std::string, std::map<K, V>> groups;
  GroupOpRegistryResults(sorted, groups);
  PrintGroupNames(groups);
  PrintIncludes(groups);
  // std::cout << "#ifndef ONEFLOW_USER_OP_GEN\n";
  // std::cout << "#define ONEFLOW_USER_OP_GEN\n\n";

  for (const auto& kv : groups) {
    auto group_name = kv.first;
    auto results = kv.second;
    std::cout << "// Group: " << group_name << "\n";
    PrintNamesInResults(results);
    std::cout << "// "
              << "Total: " << kv.second.size() << "\n\n";
    CHECK(kv.second.size()) << group_name;
    auto get_group_by_name = "GET_ONEFLOW_" + group_name + "_OP_DEFINITIONS";
    auto group_def_name = "ONEFLOW_" + group_name + "_OPS";
    std::cout << "#ifdef " << get_group_by_name << "\n\n";
    // std::cout << "#ifndef " << group_def_name << "\n\n";
    // std::cout << "#define " << group_def_name << "\n\n";
    PrintODSFromOpRegistryResults(results);
    // std::cout << "#endif // " << group_def_name << "\n\n";
    std::cout << "#endif // " << get_group_by_name << "\n\n";
  }
  of.flush();
  of.close();

  std::cout.rdbuf(coutBuf);
  return 0;
}
