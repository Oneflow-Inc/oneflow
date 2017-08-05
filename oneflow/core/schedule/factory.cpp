#include "oneflow/core/schedule/factory.h"

namespace oneflow {
namespace schedule {

std::unordered_map<std::string, std::unique_ptr<PolicyHub>>&
PolicyHubFactory() {
  static std::unordered_map<std::string, std::unique_ptr<PolicyHub>> l;
  return l;
}

}  // namespace schedule
}  // namespace oneflow
