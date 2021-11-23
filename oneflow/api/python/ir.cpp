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
#include <glog/logging.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/ir/include/OneFlow/Extension.h"
#endif  // WITH_MLIR
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace {
using namespace oneflow;
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
bool IsInvolutionOp(const std::string& op_name) {
  return GetInvolutionOps().find(op_name) != GetInvolutionOps().end();
}
bool IsIdempotentOp(const std::string& op_name) {
  return GetIdempotentOps().find(op_name) != GetIdempotentOps().end();
}
std::string GetBaseOp(const std::string& op_name) {
  if (IsInvolutionOp(op_name)) {
    return "OneFlow_InvolutionBaseOp";
  } else if (IsIdempotentOp(op_name)) {
    return "OneFlow_IdempotentBaseOp";
  } else {
    return "OneFlow_BaseOp";
  }
}

bool ShouldGenEmptyBody(const std::string& op_name) {
  if (IsInvolutionOp(op_name) || IsIdempotentOp(op_name)) {
    return true;
  } else {
    return false;
  }
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
    const ::google::protobuf::RepeatedPtrField< ::oneflow::UserOpDef_ArgDef>& arg_defs) {
  uint32_t num_variadic_op = 0;
  for (const auto& arg_def : arg_defs) {
    if (arg_def.is_optional()) { num_variadic_op += 1; }
    if (arg_def.num_as_min()) { num_variadic_op += 1; }
  }
  return num_variadic_op > 1;
}

void PrintTraitAttrs(const oneflow::UserOpDef& op_def) {
  const bool need_operand_segment_sizes = HasMultipleVariadic(op_def.input());
  const bool need_result_segment_sizes = HasMultipleVariadic(op_def.output());
  if (need_operand_segment_sizes || need_result_segment_sizes) {
    std::cout << "  let input = (ins"
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

void PrintBody(const oneflow::UserOpDef& op_def) {
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
  std::cout << "}"
            << "\n";

  // trait attrs
  PrintTraitAttrs(op_def);
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
    for (const auto& kv : sorted) {
      const oneflow::user_op::OpRegistryResult& r = kv.second;
      std::cout << "def OneFlow_" << convertToCamelFromSnakeCase(kv.first, true)
                << "Op : " << GetBaseOp(r.op_type_name) << "<\"" << kv.first << "\", []> ";
      const oneflow::UserOpDef& op_def = r.op_def;
      if (ShouldGenEmptyBody(r.op_type_name)) {
        std::cout << "{}\n";
      } else {
        PrintBody(op_def);
      }
      std::cout << "\n";
    }
  });
}

}  // namespace oneflow
