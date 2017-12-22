#include "oneflow/core/job/improver.h"

namespace oneflow {

Plan Improver::Improve(const Plan& naive_plan, const AvailableMemDesc& amd,
                       const std::string& act_event_filepath) {
  return naive_plan;  // TODO
}

}  // namespace oneflow
