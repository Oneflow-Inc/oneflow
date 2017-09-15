#include "oneflow/core/schedule/schedule_factory_configure.h"
#include "oneflow/core/schedule/allocator.h"
#include "oneflow/core/schedule/demo_sgraph.h"
#include "oneflow/core/schedule/empty_utilization_analyzer.h"
#include "oneflow/core/schedule/formula_schedule_engine.h"
#include "oneflow/core/schedule/plan_sgraph.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"
#include "oneflow/core/schedule/simulator_utilization_analyzer.h"
#include "oneflow/core/schedule/ugraph_schedule_engine.h"
#include "oneflow/core/schedule/utilization_analyzer_factory.h"

namespace oneflow {
namespace schedule {

REGISTER_SCHEDULE_FACTORY_PROVIDER("base")
    ->Set(of_make_unique<SGraphConcreteFactory<PlanSGraph>>())
    ->Set(of_make_unique<
          UtilizationAnalyzerConcreteFactory<EmptyUtilizationAnalyzer>>())
    ->Set(of_make_unique<SessionConcreteFactory<FixedBatchSession<2u>>>())
    ->Set(of_make_unique<
          ScheduleEngineConcreteFactory<NaiveFormulaScheduleEngine>>())
    ->Set(of_make_unique<ValidatorFactory>())
    ->Set(of_make_unique<AllocatorFactory>());

REGISTER_SCHEDULE_FACTORY_PROVIDER("default")->Merge(
    ScheduleFactoryConfigure::Provider("base"));

REGISTER_SCHEDULE_FACTORY_PROVIDER("utilization")
    ->Merge(ScheduleFactoryConfigure::Provider("base"))
    ->Set(of_make_unique<UtilizationAnalyzerFactory>())
    ->Set(
        of_make_unique<ScheduleEngineConcreteFactory<UGraphScheduleEngine>>());

REGISTER_SCHEDULE_FACTORY_PROVIDER("simulation_utilization")
    ->Merge(ScheduleFactoryConfigure::Provider("utilization"))
    ->Set(of_make_unique<
          UtilizationAnalyzerConcreteFactory<SimulatorUtilizationAnalyzer>>());

REGISTER_SCHEDULE_FACTORY_PROVIDER("empty_allocator")
    ->Merge(ScheduleFactoryConfigure::Provider("base"))
    ->Set(of_make_unique<AllocatorConcreteFactory<EmptyAllocator>>());

REGISTER_SCHEDULE_FACTORY_PROVIDER("demo")
    ->Merge(ScheduleFactoryConfigure::Provider("base"))
    ->Set(of_make_unique<SGraphConcreteFactory<DemoSGraph>>());

// REGISTER_SCHEDULE_FACTORY_PROVIDER("simulator_schedule_engine")
//    ->Merge(ScheduleFactoryConfigure::Provider("base"))
//    ->Set(of_make_unique<SessionFactory>())
//    ->Set(of_make_unique<
//          ScheduleEngineConcreteFactory<SimulatorScheduleEngine>>());

// REGISTER_SCHEDULE_FACTORY_PROVIDER("small_batch_num")
//    ->Merge(ScheduleFactoryConfigure::Provider("base"))
//    ->Set(of_make_unique<SessionConcreteFactory<FixedBatchSession<2u>>>());

}  // namespace schedule
}  // namespace oneflow
