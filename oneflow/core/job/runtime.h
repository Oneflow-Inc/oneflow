#ifndef ONEFLOW_CORE_JOB_RUNTIME_H_
#define ONEFLOW_CORE_JOB_RUNTIME_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

namespace extension {
class RuntimeExtensionContext;
}

class Runtime final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Runtime);
  Runtime() = delete;
  ~Runtime();

  Runtime(const Plan& plan, size_t total_piece_num, bool is_experiment_phase);

  void set_runtime_ext_ctx(std::shared_ptr<extension::RuntimeExtensionContext> new_ctx) {
    runtime_ext_ctx_ = new_ctx;
  }
  const std::shared_ptr<extension::RuntimeExtensionContext> get_runtime_ext_ctx() const {
    return runtime_ext_ctx_;
  }

 private:
  void NewAllGlobal(const Plan& plan, size_t total_piece_num, bool is_experiment_phase);
  void DeleteAllGlobal();
  std::shared_ptr<extension::RuntimeExtensionContext> runtime_ext_ctx_;
};

}  // namespace oneflow

#endif
