#ifndef ONEFLOW_CORE_SCHEDULE_SCHEDULE_FACTORY_CONFIGURE_H_
#define ONEFLOW_CORE_SCHEDULE_SCHEDULE_FACTORY_CONFIGURE_H_

#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/schedule_factory_provider.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryConfigure final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScheduleFactoryConfigure);

  static const ScheduleFactoryProvider* Provider(const std::string& name) {
    const ScheduleFactoryProvider* p = providers()[name].get();
    CHECK(p);
    return p;
  }

  static const ScheduleFactoryProvider* Default() {
    return Provider("default");
  }

  static ScheduleFactoryProvider* EnrollProvider(const std::string& name) {
    providers().emplace(name, of_make_unique<ScheduleFactoryProvider>(name));
    return providers()[name].get();
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

#define REGISTER_SCHEDULE_FACTORY_PROVIDER(name)        \
  static auto OF_PP_CAT(var_policy_hub_, __COUNTER__) = \
      ScheduleFactoryConfigure::EnrollProvider(name)

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_SCHEDULE_FACTORY_CONFIGURE_H_
