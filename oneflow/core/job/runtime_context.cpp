#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void RuntimeCtx::NewCounter(const std::string& name, int64_t val) {
  LOG(INFO) << "NewCounter " << name << " " << val;
  CHECK(counters_.emplace(name, of_make_unique<BlockingCounter>(val)).second);
}

void RuntimeCtx::DecreaseCounter(const std::string& name) {
  int64_t cur_val = counters_.at(name)->Decrease();
  LOG(INFO) << "DecreaseCounter " << name << ", current val is " << cur_val;
}

void RuntimeCtx::WaitUntilCntEqualZero(const std::string& name) {
  counters_.at(name)->WaitUntilCntEqualZero();
}

RuntimeCtx::RuntimeCtx(int64_t total_piece_num, bool is_experiment_phase) {
  total_piece_num_ = total_piece_num;
  is_experiment_phase_ = is_experiment_phase;
}

}  // namespace oneflow
