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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "inja/inja.hpp"

#include <iomanip>
#include <string>

using namespace llvm;
using inja::json;

namespace oneflow {
namespace tblgen {

cl::OptionCategory opSchemaCat("Options for -gen-op-schema");

cl::opt<std::string> sourceIncludeFilename{
    "op-include", cl::desc("header filename to include in source file"),
    cl::value_desc("include filename"), cl::init(""), cl::cat(opSchemaCat)};

cl::opt<std::string> dumpJson{"op-dump-json",
                              cl::desc("dump tablegen code to json in provided file"),
                              cl::value_desc("filename"), cl::init(""), cl::cat(opSchemaCat)};

enum class FileTarget {
  kHeader = 1,
  kSource,
};

template<FileTarget Target>
class OpSchemaEmitter {
 public:
  explicit OpSchemaEmitter(RecordKeeper& RK);

  void run(raw_ostream& os);

  void emitInputAndOutput(const Record* def, json* op) const;

  void emitAttrs(const Record* def, json* op) const;

  void emitInt(const Record* def, StringRef fieldname, json* op) const;
  void emitBit(const Record* def, StringRef fieldname, json* op) const;
  void emitTrait(const Record* def, StringRef fieldname, StringRef traitname, json* op) const;

 private:
  static std::string emitType(const std::string& ods_type) {
#define OP_SCHEMA(ods, cpp) \
  if (ods_type == #ods) return #cpp;
#include "op_schema_types.inc"
#undef OP_SCHEMA
    PrintFatalError("undefined attribute type: " + ods_type);
  }

 private:
  RecordKeeper& records;

  StringRef op_type_name;
  StringRef op_name;

  inja::Environment env;
  inja::Template temp;
  static const std::string code;
};

template<FileTarget Target>
OpSchemaEmitter<Target>::OpSchemaEmitter(RecordKeeper& RK) : records(RK) {
  env.add_callback("quoted", 1, [](inja::Arguments& args) {
    auto str = args.at(0)->get<std::string>();
    std::ostringstream os;
    os << std::quoted(str);
    return os.str();
  });
  env.add_callback("to_header", 1, [](inja::Arguments& args) {
    auto str = args.at(0)->get<std::string>();
    auto dot_pos = str.find_last_of('.');
    if (dot_pos != std::string::npos) { str.replace(dot_pos, str.size() - dot_pos, ".h"); }

    // assume that the source and header file is in the same directory
    auto slash_pos = str.find_last_of('/');
    if (slash_pos != std::string::npos) { str.replace(0, slash_pos + 1, ""); }
    return str;
  });
  temp = env.parse(code);
}

template<FileTarget Target>
void OpSchemaEmitter<Target>::run(raw_ostream& os) {
  emitSourceFileHeader("oneflow op schema", os);
  json ops = json::object();

  for (const auto& def : records.getAllDerivedDefinitions("OneFlow_BaseOp")) {
    op_type_name = def->getValueAsString("opName");
    if (op_type_name.empty()) {
      PrintFatalError(def, "`opName` of op definitions cannot be omitted");
    }
    op_name = def->getName();
    if (!op_name.consume_front("OneFlow_")) {
      PrintFatalError(def, "op name is not start with `OneFlow_`: " + op_name.str());
    }
    json op{{"name", op_type_name},
            {"input", json::array()},
            {"output", json::array()},
            {"attrs", json::array()}};

    emitInputAndOutput(def, &op);
    emitAttrs(def, &op);
    emitInt(def, "same_output_regst_num", &op);
    emitTrait(def, "no_grad", "NoGrad", &op);
    emitTrait(def, "support_non_contiguous", "SupportNonContiguous", &op);
    emitTrait(def, "cpu_only", "CpuOnly", &op);
    emitBit(def, "has_nd_sbp_infer_fn", &op);
    emitBit(def, "has_get_sbp_fn", &op);
    emitBit(def, "has_logical_tensor_desc_infer_fn", &op);
    emitBit(def, "has_physical_tensor_desc_infer_fn", &op);
    emitBit(def, "has_data_type_infer_fn", &op);
    emitBit(def, "has_device_and_stream_infer_fn", &op);
    emitBit(def, "has_input_arg_modify_fn", &op);
    emitBit(def, "has_output_arg_modify_fn", &op);
    emitBit(def, "has_output_blob_time_shape_infer_fn", &op);
    emitBit(def, "has_sbp_signature_infer_fn", &op);
    emitBit(def, "has_get_nd_sbp_fn", &op);
    emitBit(def, "has_enumerate_nd_sbp_signatures_fn", &op);
    emitBit(def, "has_dump_nd_sbp_signature_for_op_conf_fn", &op);
    emitBit(def, "has_compute_complexity_fn", &op);
    emitBit(def, "has_check_fn", &op);
    ops[op_name.str()] = op;
  }

  auto* option = static_cast<cl::opt<std::string>*>(cl::getRegisteredOptions().lookup("o"));
  auto filename = option->getValue();
  filename = filename != "-" ? filename : "";
  json data{{"filename", filename}, {"ops", ops}};

  if (Target == FileTarget::kSource) { data["include"] = sourceIncludeFilename.getValue(); }
  if (!dumpJson.empty()) {
    std::ofstream file(dumpJson);
    file << data.dump();
  }
  os << env.render(temp, data);
}

template<FileTarget Target>
void OpSchemaEmitter<Target>::emitInputAndOutput(const Record* def, json* op) const {
  const auto* input = def->getValueAsDag("input");
  for (size_t i = 0; i < input->getNumArgs(); ++i) {
    const auto* A = dyn_cast<DefInit>(input->getArg(i))->getDef();
    bool is_optional = A->isSubClassOf("Optional");
    auto NS = input->getArgName(i)->getAsUnquotedString();
    (*op)["input"].push_back({{"name", NS}, {"is_optional", is_optional}, {"size", 1}});
  }
  const auto* output = def->getValueAsDag("output");
  for (size_t i = 0; i < output->getNumArgs(); ++i) {
    const auto* A = dyn_cast<DefInit>(output->getArg(i))->getDef();
    bool is_optional = A->isSubClassOf("Optional");
    auto NS = output->getArgName(i)->getAsUnquotedString();
    (*op)["output"].push_back({{"name", NS}, {"is_optional", is_optional}, {"size", 1}});
  }
}

template<FileTarget Target>
void OpSchemaEmitter<Target>::emitAttrs(const Record* def, json* op) const {
  const auto* attrs = def->getValueAsDag("attrs");
  for (size_t i = 0; i < attrs->getNumArgs(); ++i) {
    const auto* A = dyn_cast<DefInit>(attrs->getArg(i))->getDef();
    std::string AS;
    if (!A->isAnonymous()) {
      AS = A->getNameInitAsString();
    } else {
      AS = A->getValueAsDef("baseAttr")->getNameInitAsString();
    }
    auto NS = attrs->getArgName(i)->getAsUnquotedString();
    // FlatSymbolRefAttr:$callee,
    if ("callee" == NS && "FlatSymbolRefAttr" == AS) { continue; }
    json attr{{"name", NS}, {"type", emitType(AS)}};

    if (auto DV = A->getValueAsOptionalString("defaultValue")) { attr["default"] = DV.getValue(); }

    (*op)["attrs"].push_back(attr);
  }
}

template<FileTarget Target>
void OpSchemaEmitter<Target>::emitBit(const Record* def, StringRef fieldname, json* op) const {
  (*op)[fieldname.str()] = def->getValueAsBit(fieldname);
}

template<FileTarget Target>
void OpSchemaEmitter<Target>::emitTrait(const Record* def, StringRef fieldname, StringRef traitname,
                                        json* op) const {
  bool hasTrait = false;

  for (auto elem : *def->getValueAsListInit("traits")) {
    if (elem->getAsString() == traitname) {
      hasTrait = true;
      break;
    }
  }

  (*op)[fieldname.str()] = hasTrait;
}

template<FileTarget Target>
void OpSchemaEmitter<Target>::emitInt(const Record* def, StringRef fieldname, json* op) const {
  (*op)[fieldname.str()] = def->getValueAsInt(fieldname);
}

template<>
const std::string OpSchemaEmitter<FileTarget::kHeader>::code{
#include "op_schema_header.inc"
};

template<>
const std::string OpSchemaEmitter<FileTarget::kSource>::code{
#include "op_schema_source.inc"
};

void EmitOpSchemaHeader(RecordKeeper& RK, raw_ostream& os) {
  OpSchemaEmitter<FileTarget::kHeader>(RK).run(os);
}

void EmitOpSchemaSource(RecordKeeper& RK, raw_ostream& os) {
  OpSchemaEmitter<FileTarget::kSource>(RK).run(os);
}

}  // namespace tblgen
}  // namespace oneflow
