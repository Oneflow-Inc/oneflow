#ifndef ONEFLOW_CORE_JOB_COMPLETER_REGISTRY_H
#define ONEFLOW_CORE_JOB_COMPLETER_REGISTRY_H

#include "oneflow/core/job_completer/optimizer_base.h"

namespace oneflow {

static HashMap<std::string, OptimizerBase*> name2optimizer;
class OptimizerRegistry {
 public:
  static OptimizerBase* Lookup(std::string name) {
    if (name2optimizer.empty()) { LOG(FATAL) << "no optimizer registered"; }
    const auto it = name2optimizer.find(name);
    if (it == name2optimizer.end()) {
      std::string all_registered_name;
      for (const auto tuple : name2optimizer) {
        all_registered_name += ", ";
        all_registered_name += tuple.first;
      }
      LOG(FATAL) << "optimizer " << name << " not found, all: " << all_registered_name;
    }
    return it->second;
  }
  static Maybe<void> Register(std::string name, OptimizerBase* optimizer) {
    CHECK_OR_RETURN(name2optimizer.find(name) == name2optimizer.end())
        << "optmizer " << name << " registered";
    CHECK(name2optimizer.emplace(name, optimizer).second);
    return Maybe<void>::Ok();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_REGISTRY_H
