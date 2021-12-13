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
#include <algorithm>
#include <iomanip>
#include <string>
#include <vector>

#include "inja.hpp"

using namespace llvm;

namespace oneflow {
namespace tblgen {

cl::OptionCategory opSchemaCat("Options for -gen-op-schema");

cl::opt<std::string> sourceIncludeFilename{
    "op-include", cl::desc("header filename to include in source file"),
    cl::value_desc("include filename"), cl::init(""), cl::cat(opSchemaCat)};

cl::opt<std::string> dumpJson{"op-dump-json",
                              cl::desc("dump tablegen code to json in provided file"),
                              cl::value_desc("filename"), cl::init(""), cl::cat(opSchemaCat)};

const std::string& outputFilename() {
  auto* option =
      static_cast<cl::opt<std::string>*>(cl::getRegisteredOptions().lookup("o"));  // NOLINT
  return option->getValue();
}

enum class OpSchemaTarget {
  Header = 1,
  Source,
};

template<OpSchemaTarget Target>
class OpSchemaEmitter {
 public:
  explicit OpSchemaEmitter(RecordKeeper& RK);

  struct CustomCode {
    bool is_member;
    StringRef code;
  };
  void run(raw_ostream& os);
  bool getCustomCode(const Record* def, StringRef fieldname, CustomCode* code);

 private:
  static nonstd::string_view toCppType(nonstd::string_view ods_type) {
#define OP_SCHEMA(ods, cpp) \
  if (ods_type == #ods) return #cpp;
#include "op_schema_types.inc"
#undef OP_SCHEMA
    PrintFatalError("undefined attribute type: " + (std::string)ods_type);
  }

  inja::Environment env;
  inja::Template temp;
  RecordKeeper& records;

  static const nonstd::string_view code;
};

template<OpSchemaTarget Target>
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

template<OpSchemaTarget Target>
void OpSchemaEmitter<Target>::run(raw_ostream& os) {
  using inja::json;
  emitSourceFileHeader("oneflow op schema", os);
  json ops = json::object();
  for (const auto& def : records.getAllDerivedDefinitions("OneFlow_BaseOp")) {
    auto name = def->getValueAsString("opName");
    if (name.empty()) {  // NOLINT
      PrintFatalError(def, "`opName` of op definitions cannot be omitted");
    }
    json op{{"name", name}, {"attrs", json::object()}};

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
      op["attrs"][NS] = {{"type", toCppType(AS)}};

      if (auto DV = A->getValueAsOptionalString("defaultValue")) {
        op["attrs"][NS]["default"] = DV.getValue();
      }
    }

    CustomCode code;
    if (getCustomCode(def, "infer_nd_sbp", &code)) { std::cout << code.code.str() << std::endl; }

    auto op_name = def->getName();
    if (!op_name.consume_front("OneFlow_")) {
      PrintFatalError(def, "op name is not start with `OneFlow_`: " + op_name.str());
    }
    ops[op_name.str()] = op;
  }
  if (ops.empty()) { PrintWarning("no `Op` in this file"); }

  auto filename = outputFilename();
  filename = filename != "-" ? filename : "";

  json data{{"filename", filename}, {"ops", ops}};

  if (Target == OpSchemaTarget::Source) { data["include"] = sourceIncludeFilename; }

  if (!dumpJson.empty()) {
    std::ofstream file(dumpJson);
    file << data.dump();
  }
  os << env.render(temp, data);
}

template<OpSchemaTarget Target>
bool OpSchemaEmitter<Target>::getCustomCode(const Record* def, StringRef fieldname,
                                            CustomCode* code) {
  auto* valueInit = def->getValueInit(fieldname);
  StringInit* stringInit = dyn_cast<StringInit>(valueInit);
  if (!stringInit || stringInit->getValue().empty()) { return false; }
  auto value = stringInit->getValue().ltrim().rtrim(" \t\v\f\r");
  if (!value.consume_front("return ")) {
    PrintFatalError(
        def, "Invalid " + fieldname.str() + " code (note: should start with return identifier)");
  }
  size_t quto_start_pos = value.find_first_of("(");
  size_t quto_end_pos = value.find_last_of(")");
  if (quto_start_pos == std::string::npos || quto_end_pos == std::string::npos
      || quto_start_pos > quto_end_pos) {
    PrintFatalError(def, "Invalid " + fieldname.str() + " code (note: missing brackets)");
  }
  value = value.substr(0, quto_start_pos).ltrim().rtrim(" \t\v\f\r");
  if (value.consume_front("::")) {
    code->is_member = true;
  } else {
    code->is_member = false;
  }
  code->code = value;
  return true;
}

template<>
const nonstd::string_view OpSchemaEmitter<OpSchemaTarget::Header>::code{
#include "op_schema_header.inc"
};

template<>
const nonstd::string_view OpSchemaEmitter<OpSchemaTarget::Source>::code{
#include "op_schema_source.inc"
};

void EmitOpSchemaHeader(RecordKeeper& RK, raw_ostream& os) {
  OpSchemaEmitter<OpSchemaTarget::Header>(RK).run(os);
}

void EmitOpSchemaSource(RecordKeeper& RK, raw_ostream& os) {
  OpSchemaEmitter<OpSchemaTarget::Source>(RK).run(os);
}

}  // namespace tblgen
}  // namespace oneflow
