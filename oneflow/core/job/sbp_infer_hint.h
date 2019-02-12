#ifndef ONEFLOW_CORE_JOB_SBP_INFER_HINT_H_
#define ONEFLOW_CORE_JOB_SBP_INFER_HINT_H_

#include "oneflow/core/job/sbp_infer_hint_conf.pb.h"

namespace oneflow {

class SbpInferHint final {
 public:
  SbpInferHint() : parallel_num_(-1), num_axes_(-1) {}
  SbpInferHint(const SbpInferHint&) = default;
  ~SbpInferHint() = default;

  // Getters
  int64_t parallel_num() const;
  int64_t num_axes() const;
  const SplitParallel& model_split() const;
  const BroadcastParallel& model_clone() const;
  const SplitParallel& data_split() const;
  const PartialSumParallel& data_partial_sum() const;
  bool has_model_split() const;
  bool has_model_clone() const;
  bool is_data_split() const;
  bool has_data_partial_sum() const;
  bool is_model_blob() const { return sbp_infer_hint_conf_.is_model_blob(); }
  bool is_data_blob() const { return !is_model_blob(); }
  const SbpParallel& sbp_parallel() const { return sbp_infer_hint_conf_.sbp_parallel(); }

  // Setters
  void set_parallel_num(int64_t val) { parallel_num_ = val; }
  void set_num_axes(int64_t val) { num_axes_ = val; }
  SplitParallel* mutable_model_split();
  BroadcastParallel* mutable_model_clone();
  SplitParallel* mutable_data_split();
  PartialSumParallel* mutable_data_partial_sum();

 private:
  int64_t parallel_num_;
  int64_t num_axes_;
  SbpInferHintConf sbp_infer_hint_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SBP_INFER_HINT_H_
