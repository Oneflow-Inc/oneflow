#include "oneflow/core/schedule/factory.h"
#include "oneflow/core/schedule/naive.h"
#include "oneflow/core/schedule/simulator_session.h"
#include "oneflow/core/schedule/validator.h"

namespace oneflow {
namespace schedule {

REGISTER_POLICY_HUB("simulator-policy")
    ->Add(unique_ptr_new<ScheduleEngineSimulatorPolicy>())
    ->Add(unique_ptr_new<RetimingSimulatorPolicy>())
    ->Add(unique_ptr_new<AllocatorSimulatorPolicy>())
    ->Add(unique_ptr_new<AllocationValidatorSimplePolicy>());

REGISTER_POLICY_HUB("naive")
    ->Merge(PH("simulator-policy"))
    ->Add(unique_ptr_new<TestGraphGeneratorNaivePolicy>())
    ->Add(unique_ptr_new<PrinterNaivePolicy>());

}  // namespace schedule
}  // namespace oneflow
