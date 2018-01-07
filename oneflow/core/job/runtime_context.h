#ifndef ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
#define ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_

#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class RuntimeCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RuntimeCtx);
  RuntimeCtx() = delete;
  ~RuntimeCtx() = default;

  OF_SINGLETON(RuntimeCtx);

  int64_t total_piece_num() const { return total_piece_num_; }
  bool is_experiment_phase() const { return is_experiment_phase_; }

  void NewCounter(const std::string& name, int64_t val);
  void DecreaseCounter(const std::string& name);
  void WaitUntilCntEqualZero(const std::string& name);

 private:
  RuntimeCtx(int64_t total_piece_num, bool is_experiment_phase);

  int64_t total_piece_num_;
  bool is_experiment_phase_;
  HashMap<std::string, std::unique_ptr<BlockingCounter>> counters_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
