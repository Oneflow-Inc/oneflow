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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/SetTheory.h"

#include "backends.h"

using namespace llvm;
using namespace oneflow::tblgen;

enum ActionType {
  PrintRecords,
  PrintDetailedRecords,
  NullBackend,
  DumpJSON,
  PrintEnums,
  PrintSets,
  GenOpSchemaHeader,
  GenOpSchemaSource,
};

namespace llvm {
cl::opt<bool> EmitLongStrLiterals(
    "long-string-literals",
    cl::desc("when emitting large string tables, prefer string literals over "
             "comma-separated char literals. This can be a readability and "
             "compile-time performance win, but upsets some compilers"),
    cl::Hidden, cl::init(true));
}  // end namespace llvm

namespace {
cl::opt<ActionType> Action(
    cl::desc("Action to perform:"),
    cl::values(clEnumValN(PrintRecords, "print-records", "Print all records to stdout (default)"),
               clEnumValN(PrintDetailedRecords, "print-detailed-records",
                          "Print full details of all records to stdout"),
               clEnumValN(NullBackend, "null-backend",
                          "Do nothing after parsing (useful for timing)"),
               clEnumValN(DumpJSON, "dump-json", "Dump all records as machine-readable JSON"),
               clEnumValN(PrintEnums, "print-enums", "Print enum values for a class"),
               clEnumValN(PrintSets, "print-sets", "Print expanded sets for testing DAG exprs"),
               clEnumValN(GenOpSchemaHeader, "gen-op-schema-h",
                          "Generate oneflow op schema header code (.h)"),
               clEnumValN(GenOpSchemaSource, "gen-op-schema-cpp",
                          "Generate oneflow op schema source code (.cpp)")));

cl::OptionCategory PrintEnumsCat("Options for -print-enums");
cl::opt<std::string> Class("class", cl::desc("Print Enum list for this class"),
                           cl::value_desc("class name"), cl::cat(PrintEnumsCat));

bool LLVMTableGenMain(raw_ostream& OS, RecordKeeper& Records) {
  switch (Action) {
    case PrintRecords: OS << Records; break;
    case PrintDetailedRecords: EmitDetailedRecords(Records, OS); break;
    case NullBackend: break;
    case DumpJSON: EmitJSON(Records, OS); break;
    case PrintEnums: {
      for (Record* Rec : Records.getAllDerivedDefinitions(Class)) OS << Rec->getName() << ", ";
      OS << "\n";
      break;
    }
    case PrintSets: {
      SetTheory Sets;
      Sets.addFieldExpander("Set", "Elements");
      for (Record* Rec : Records.getAllDerivedDefinitions("Set")) {
        OS << Rec->getName() << " = [";
        const std::vector<Record*>* Elts = Sets.expand(Rec);
        assert(Elts && "Couldn't expand Set instance");
        for (Record* Elt : *Elts) OS << ' ' << Elt->getName();
        OS << " ]\n";
      }
      break;
    }
    case GenOpSchemaHeader: EmitOpSchemaHeader(Records, OS); break;
    case GenOpSchemaSource: EmitOpSchemaSource(Records, OS); break;
  }

  return false;
}
}  // namespace

int main(int argc, char** argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  return TableGenMain(argv[0], &LLVMTableGenMain);
}
