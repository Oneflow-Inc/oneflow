#ifndef ONEFLOW_CORE_JOB_SBP_INFER_HINT_H_
#define ONEFLOW_CORE_JOB_SBP_INFER_HINT_H_

#include "oneflow/core/job/sbp_infer_hint_conf.pb.h"

namespace oneflow {

class SbpInferHint final {
 public:
  SbpInferHint() : is_model_blob_(false), parallel_num_(-1), num_axes_(-1), split_axis_(-1) {}
  SbpInferHint(bool is_model_blob, int64_t parallel_num, int64_t num_axes, int64_t split_axis)
      : is_model_blob_(is_model_blob),
        parallel_num_(parallel_num),
        num_axes_(num_axes),
        split_axis_(split_axis) {}
  SbpInferHint(const SbpInferHint&) = default;
  ~SbpInferHint() = default;

  // Getters
  int64_t parallel_num() const;
  int64_t num_axes() const;
  int64_t split_axis() const;
  bool is_model_split() const;
  bool is_model_broadcast() const;
  bool is_data_split() const;
  bool is_data_partial_sum() const;
  bool is_model_blob() const { return sbp_infer_hint_conf_.is_model_blob(); }
  bool is_data_blob() const { return !is_model_blob(); }

  // Setters
  void set_is_model_blob(bool val) { is_model_blob_ = val; }
  void set_parallel_num(int64_t val) { parallel_num_ = val; }
  void set_num_axes(int64_t val) { num_axes_ = val; }
  void set_split_axis(int64_t val) { split_axis_ = val; }
  SplitParallel* mutable_model_split();
  BroadcastParallel* mutable_model_clone();
  SplitParallel* mutable_data_split();
  PartialSumParallel* mutable_data_partial_sum();

 private:
  bool is_model_blob_;
  int64_t parallel_num_;
  int64_t num_axes_;
  int64_t split_axis_;
  SbpInferHintConf sbp_infer_hint_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SBP_INFER_HINT_H_
