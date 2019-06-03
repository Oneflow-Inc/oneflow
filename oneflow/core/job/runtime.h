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

  Runtime(const Plan& plan, size_t total_piece_num, bool is_experiment_phase);

 private:
  void NewAllGlobal(const Plan& plan, size_t total_piece_num, bool is_experiment_phase);
  void DeleteAllGlobal();
};

}  // namespace oneflow

#endif
