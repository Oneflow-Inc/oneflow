#ifndef ONEFLOW_CORE_JOB_IMPROVER_H_
#define ONEFLOW_CORE_JOB_IMPROVER_H_

#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

class Improver final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Improver);
  ~Improver() = default;

  OF_SINGLETON(Improver);

  Plan Improve(const Plan& naive_plan, const std::string& act_event_filepath);
  size_t MemorySize(int64_t machine_id, int64_t memory_zone_id) const;
  int64_t GetMemoryZoneId(const MemoryCase& mem_case) const;

 private:
  Improver() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_IMPROVER_H_
