#ifndef ONEFLOW_CORE_SCHEDULE_FACTORY_PROVIDER_H_
#define ONEFLOW_CORE_SCHEDULE_FACTORY_PROVIDER_H_

#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/allocator_factory.h"
#include "oneflow/core/schedule/schedule_engine_factory.h"
#include "oneflow/core/schedule/session_factory.h"
#include "oneflow/core/schedule/sgraph_factory.h"
#include "oneflow/core/schedule/utilization_analyzer_factory.h"
#include "oneflow/core/schedule/validator_factory.h"

#define SCHEDULE_FACTORY_SEQ                                           \
  OF_PP_MAKE_TUPLE_SEQ(SGraphFactory, sgraph_factory)                  \
  OF_PP_MAKE_TUPLE_SEQ(SessionFactory, session_factory)                \
  OF_PP_MAKE_TUPLE_SEQ(ScheduleEngineFactory, schedule_engine_factory) \
  OF_PP_MAKE_TUPLE_SEQ(ValidatorFactory, validator_factory)            \
  OF_PP_MAKE_TUPLE_SEQ(AllocatorFactory, allocator_factory)            \
  OF_PP_MAKE_TUPLE_SEQ(UtilizationAnalyzerFactory, utilization_analyzer_factory)

namespace oneflow {
namespace schedule {

class ScheduleFactoryProvider final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScheduleFactoryProvider);
  explicit ScheduleFactoryProvider(const std::string name) : name_(name) {}

  ScheduleFactoryProvider* Merge(const ScheduleFactoryProvider* sfp) {
#define PROVIDER_FACTORY_CLONE(class_name, field) CLONE_FACTORY(sfp, field());
    OF_PP_FOR_EACH_TUPLE(PROVIDER_FACTORY_CLONE, SCHEDULE_FACTORY_SEQ);
    return this;
  }

#define PROVIDER_FACTORY_SET(class_name, field)                         \
  ScheduleFactoryProvider* Set(std::unique_ptr<class_name>&& factory) { \
    OF_PP_CAT(field, _) = factory->Clone(this);                         \
    return this;                                                        \
  }
  OF_PP_FOR_EACH_TUPLE(PROVIDER_FACTORY_SET, SCHEDULE_FACTORY_SEQ);

//	getter
#define PROVIDER_FACTORY_GETTER(class_name, field) \
  inline const class_name& field() const {         \
    CHECK(OF_PP_CAT(field, _).get());              \
    return *OF_PP_CAT(field, _).get();             \
  }
  OF_PP_FOR_EACH_TUPLE(PROVIDER_FACTORY_GETTER, SCHEDULE_FACTORY_SEQ);

 private:
//	setter
#define PROVIDER_FACTORY_PRIVATE_SETTER(class_name, field)       \
  inline std::unique_ptr<class_name>& OF_PP_CAT(mut_, field)() { \
    return OF_PP_CAT(field, _);                                  \
  }
  OF_PP_FOR_EACH_TUPLE(PROVIDER_FACTORY_PRIVATE_SETTER, SCHEDULE_FACTORY_SEQ);

  std::string name_;
#define PROVIDER_FACTORY_MEMBER(class_name, field) \
  std::unique_ptr<class_name> OF_PP_CAT(field, _);
  OF_PP_FOR_EACH_TUPLE(PROVIDER_FACTORY_MEMBER, SCHEDULE_FACTORY_SEQ);
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_FACTORY_PROVIDER_H_
