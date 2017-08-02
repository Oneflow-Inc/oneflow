#ifndef ONEFLOW_CORE_SCHEDULE_POLICY_FACTORY_FACTORY_H_
#define ONEFLOW_CORE_SCHEDULE_POLICY_FACTORY_FACTORY_H_

#include "oneflow/core/schedule/interface/policy.h"
#include "oneflow/core/schedule/interface/policy_hub.h"

namespace oneflow {
namespace schedule {

std::unordered_map<std::string, std::unique_ptr<PolicyHub>>& PolicyHubFactory();

inline const PolicyHub* PH(const std::string& name) {
  return PolicyHubFactory()[name].get();
}

inline PolicyHub* mut_PH(const std::string& name) {
  return PolicyHubFactory()[name].get();
}

#define REGISTER_POLICY_HUB(name)                                        \
  static auto MACRO_CONCAT(var_policy_hub_, __COUNTER__) =               \
      PolicyHubFactory().emplace(name, unique_ptr_new<PolicyHub>(name)); \
  static auto MACRO_CONCAT(var_policy_hub_, __COUNTER__) = mut_PH(name)

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_POLICY_FACTORY_FACTORY_H_
