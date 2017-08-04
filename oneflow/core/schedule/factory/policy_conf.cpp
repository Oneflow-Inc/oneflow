#include "oneflow/core/schedule/factory/factory.h"
#include "oneflow/core/schedule/implement/naive.h"
#include "oneflow/core/schedule/implement/simulator.h"
#include "oneflow/core/schedule/implement/validator.h"

namespace oneflow {
namespace schedule {

REGISTER_POLICY_HUB("simulator-policy")
    ->Add(unique_ptr_new<StaticSchedulerSimulatorPolicy>())
    ->Add(unique_ptr_new<RetimingSimulatorPolicy>())
    ->Add(unique_ptr_new<AllocatorSimulatorPolicy>())
    ->Add(unique_ptr_new<AllocationValidatorSimplePolicy>());

REGISTER_POLICY_HUB("naive")
    ->Merge(PH("simulator-policy"))
    ->Add(unique_ptr_new<TestGraphGeneratorNaivePolicy>())
    ->Add(unique_ptr_new<PrinterNaivePolicy>());

}  // namespace schedule
}  // namespace oneflow
