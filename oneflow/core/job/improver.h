#ifndef ONEFLOW_CORE_JOB_IMPROVER_H_
#define ONEFLOW_CORE_JOB_IMPROVER_H_

#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/available_memory_desc.pb.h"

namespace oneflow {

class Improver final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Improver);
  Improver() = delete;
  ~Improver() = default;

  OF_SINGLETON(Improver);

  Plan Improve(const Plan& naive_plan, const std::string& act_event_filepath);

 private:
  Improver(const AvailableMemDesc& amd);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_IMPROVER_H_
