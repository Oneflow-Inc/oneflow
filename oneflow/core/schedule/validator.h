#ifndef ONEFLOW_CORE_SCHEDULE_IMPLEMENT_VALIDATOR_H_
#define ONEFLOW_CORE_SCHEDULE_IMPLEMENT_VALIDATOR_H_

#include "oneflow/core/schedule/policy.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

class AllocationValidatorSimplePolicy : public AllocationValidatorPolicy {
 public:
  POLICY_IMPLEMENT_BOILERPLATE(AllocationValidatorSimplePolicy,
                               AllocationValidatorPolicy);

  virtual bool ValidateAllocation(const Session& session,
                                  const ScheduleResult& result);
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_IMPLEMENT_VALIDATOR_H_
