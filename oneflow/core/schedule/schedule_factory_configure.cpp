#include "oneflow/core/schedule/schedule_factory_configure.h"
#include "oneflow/core/schedule/demo_sgraph.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"

namespace oneflow {
namespace schedule {

REGISTER_SCHEDULE_FACTORY_PROVIDER("base")
    ->Set(unique_ptr_new<SGraphConcreteFactory<SGraph>>())
    ->Set(unique_ptr_new<SessionFactory>())
    ->Set(unique_ptr_new<
          ScheduleEngineConcreteFactory<SimulatorScheduleEngine>>())
    ->Set(unique_ptr_new<ValidatorFactory>())
    ->Set(unique_ptr_new<AllocatorFactory>());

REGISTER_SCHEDULE_FACTORY_PROVIDER("default")->Merge(
    ScheduleFactoryConfigure::Provider("base"));

REGISTER_SCHEDULE_FACTORY_PROVIDER("demo")
    ->Merge(ScheduleFactoryConfigure::Provider("base"))
    ->Set(unique_ptr_new<SGraphConcreteFactory<DemoSGraph>>());

}  // namespace schedule
}  // namespace oneflow
