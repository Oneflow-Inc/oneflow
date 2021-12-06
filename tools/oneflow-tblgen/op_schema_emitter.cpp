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
#include <set>
#include <string>
#include <vector>

#include "external/inja.hpp"

#define DEBUG_TYPE "op-schema-emitter"

using namespace llvm;
using namespace nlohmann;
using namespace inja;

using namespace std::string_literals;
using namespace nonstd::string_view_literals;

namespace oneflow {
namespace tblgen {

cl::opt<std::string> SourceIncludeFilename{
    "op-include", cl::desc("header filename to include in source file (used in gen-op-schema)"),
    cl::value_desc("include filename"), cl::init("")};

cl::opt<std::string> DumpJson{
    "op-dump-json", cl::desc("dump tablegen code to json in provided file (used in gen-op-schema)"),
    cl::value_desc("filename"), cl::init("")};

cl::opt<bool> HeaderOnly{"op-header-only",
                         cl::desc("only generate header files, use static inlined member functions "
                                  "instead of static fields (used in gen-op-schema)"),
                         cl::init(false)};

const std::string& outputFilename() {
  auto* Option =
      static_cast<cl::opt<std::string>*>(cl::getRegisteredOptions().lookup("o"));  // NOLINT

  return Option->getValue();
}

enum class OpSchemaTarget {
  Header = 1,
  Source,
};

template<OpSchemaTarget Target>
class OpSchemaEmitter {
 private:
  RecordKeeper& Records;
  Environment Env;
  Template Temp;

  static const nonstd::string_view Code;

  static nonstd::string_view ToCppType(nonstd::string_view OdsType) {
#define OP_SCHEMA(ods, cpp) \
  if (OdsType == #ods) return #cpp;
#include "op_schema_types.inc"
#undef OP_SCHEMA

    PrintFatalError("undefined attribute type: "s + (std::string)OdsType);
  }

 public:
  explicit OpSchemaEmitter(RecordKeeper& RK) : Records(RK) {
    Env.add_callback("quoted", 1, [](Arguments& Args) {
      auto Str = Args.at(0)->get<std::string>();
      std::ostringstream OS;
      OS << std::quoted(Str);
      return OS.str();
    });

    Env.add_callback("to_header", 1, [](Arguments& Args) {
      auto Str = Args.at(0)->get<std::string>();
      auto DotPos = Str.find_last_of('.');
      if (DotPos != std::string::npos) { Str.replace(DotPos, Str.size() - DotPos, ".h"); }

      // assume that the source and header file is in the same directory
      auto SlashPos = Str.find_last_of('/');
      if (SlashPos != std::string::npos) { Str.replace(0, SlashPos + 1, ""); }
      return Str;
    });

    Temp = Env.parse(Code);
  }

  void run(raw_ostream& OS) {
    emitSourceFileHeader("oneflow op schema", OS);

    json Ops;

    for (const auto& R : Records.getAllDerivedDefinitions("OneFlow_BaseOp")) {
      auto Name = R->getValueAsString("opName");
      if (Name == "") { PrintFatalError(R, "`opName` of op definitions cannot be omitted"); }

      json Op{{"name", Name}};

      const auto* D = R->getValueAsDag("attrs");
      if (!D) { PrintFatalError(R, "`attrs` in op should be typed as `dag`"); }

      for (size_t I = 0; I < D->getNumArgs(); ++I) {
        const auto* A = dyn_cast<DefInit>(D->getArg(I))->getDef();
        std::string AS;
        if (!A->isAnonymous()) {
          AS = A->getNameInitAsString();
        } else {
          AS = A->getValueAsDef("baseAttr")->getNameInitAsString();
        }

        auto NS = D->getArgName(I)->getAsUnquotedString();
        Op["attrs"][NS] = {{"type", ToCppType(AS)}};

        if (auto DV = A->getValueAsOptionalString("defaultValue")) {
          Op["attrs"][NS]["default"] = DV.getValue();
        }
      }

      auto OpName = R->getName();
      if (!OpName.startswith("OneFlow_")) {
        PrintFatalError("op name is not start with `OneFlow_`: " + (std::string)OpName);
      }
      Ops[(std::string)OpName.substr(nonstd::string_view("OneFlow_").length())] = Op;
    }

    if (Ops.empty()) { PrintWarning("no `Op` in this file"); }

    auto Filename = outputFilename();
    Filename = Filename != "-" ? Filename : "";

    json Data{{"filename", Filename}, {"ops", Ops}, {"header_only", HeaderOnly.getValue()}};

    if (Target == OpSchemaTarget::Source) { Data["include"] = SourceIncludeFilename; }

    if (!DumpJson.empty()) {
      std::ofstream File(DumpJson);
      File << Data.dump();
    }

    OS << Env.render(Temp, Data);
  }
};

template<>
const nonstd::string_view OpSchemaEmitter<OpSchemaTarget::Header>::Code{
#include "op_schema_header.inc"
};

template<>
const nonstd::string_view OpSchemaEmitter<OpSchemaTarget::Source>::Code{
#include "op_schema_source.inc"
};

void EmitOpSchemaHeader(RecordKeeper& RK, raw_ostream& OS) {
  OpSchemaEmitter<OpSchemaTarget::Header>(RK).run(OS);
}

void EmitOpSchemaSource(RecordKeeper& RK, raw_ostream& OS) {
  OpSchemaEmitter<OpSchemaTarget::Source>(RK).run(OS);
}

}  // namespace tblgen
}  // namespace oneflow
