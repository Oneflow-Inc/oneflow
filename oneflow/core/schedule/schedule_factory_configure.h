#ifndef ONEFLOW_CORE_SCHEDULE_SCHEDULE_FACTORY_CONFIGURE_H_
#define ONEFLOW_CORE_SCHEDULE_SCHEDULE_FACTORY_CONFIGURE_H_

#include "oneflow/core/schedule/schedule_factory_provider.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryConfigure final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScheduleFactoryConfigure);

  static ScheduleFactoryProvider* Provider(const std::string& name) {
    auto p = providers()[name].get();
    CHECK(p);
    return p;
  }

  static ScheduleFactoryProvider* Default() { return Provider("default"); }

  static ScheduleFactoryProvider* EnrollProvider(const std::string& name) {
    providers().emplace(name, unique_ptr_new<ScheduleFactoryProvider>(name));
    return Provider(name);
  }

  static std::unordered_map<std::string,
                            std::unique_ptr<ScheduleFactoryProvider>>&
  providers() {
    static std::unordered_map<std::string,
                              std::unique_ptr<ScheduleFactoryProvider>>
        p;
    return p;
  }
};

#define REGISTER_SCHEDULE_FACTORY_PROVIDER(name)           \
  static auto MACRO_CONCAT(var_policy_hub_, __COUNTER__) = \
      ScheduleFactoryConfigure::EnrollProvider(name)

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_SCHEDULE_FACTORY_CONFIGURE_H_
