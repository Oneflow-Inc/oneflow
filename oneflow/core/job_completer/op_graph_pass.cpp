#include "oneflow/core/job_completer/op_graph_pass.h"

namespace oneflow {

namespace {

HashMap<std::string, const OpGraphPass*>* PassName2FunctionPass() {
  static HashMap<std::string, const OpGraphPass*> pass_name2job_pass;
  return &pass_name2job_pass;
}

}  // namespace

void RegisterFunctionPass(const std::string& pass_name, const OpGraphPass* pass) {
  CHECK(PassName2FunctionPass()->emplace(pass_name, pass).second);
}

const OpGraphPass& FindFunctionPass(const std::string& pass_name) {
  const auto& iter = PassName2FunctionPass()->find(pass_name);
  CHECK(iter != PassName2FunctionPass()->end());
  return *iter->second;
}

}  // namespace oneflow
