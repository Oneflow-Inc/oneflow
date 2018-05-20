#ifndef ONEFLOW_CORE_JOB_PROFILER_H_
#define ONEFLOW_CORE_JOB_PROFILER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class Profiler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Profiler);
  Profiler() = default;
  ~Profiler() = default;

  void Profile(const Plan& plan, const std::string& act_event_filepath);

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_PROFILER_H_
