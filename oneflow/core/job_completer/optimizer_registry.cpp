#include "oneflow/core/job_completer/optimizer_registry.h"

namespace oneflow {
namespace {

HashMap<std::string, OptimizerBase*>* MutOptimizerRegistry() {
  static HashMap<std::string, OptimizerBase*> registry;
  return &registry;
}

}  // namespace

OptimizerBase* OptimizerRegistry::Lookup(std::string name) {
  if (MutOptimizerRegistry()->empty()) { LOG(FATAL) << "no optimizer registered"; }
  const auto it = MutOptimizerRegistry()->find(name);
  if (it == MutOptimizerRegistry()->end()) {
    std::string all_registered_name;
    for (const auto tuple : *MutOptimizerRegistry()) {
      all_registered_name += ", ";
      all_registered_name += tuple.first;
    }
    LOG(FATAL) << "optimizer " << name << " not found, all: " << all_registered_name;
  }
  return it->second;
}

Maybe<void> OptimizerRegistry::Register(std::string name, OptimizerBase* optimizer) {
  CHECK_OR_RETURN(MutOptimizerRegistry()->find(name) == MutOptimizerRegistry()->end())
      << "optmizer " << name << " registered";
  CHECK(MutOptimizerRegistry()->emplace(name, optimizer).second);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
