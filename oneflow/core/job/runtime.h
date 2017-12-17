#ifndef ONEFLOW_CORE_JOB_RUNTIME_H_
#define ONEFLOW_CORE_JOB_RUNTIME_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

class Runtime final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Runtime);
  Runtime() = delete;
  ~Runtime() = default;

  OF_SINGLETON(Runtime);

 private:
  Runtime(const Plan& plan, bool is_experiment_phase);

  void NewAllSingleton(const Plan& plan, bool is_experiment_phase);
  void DeleteAllSingleton();
};

}  // namespace oneflow

#endif
