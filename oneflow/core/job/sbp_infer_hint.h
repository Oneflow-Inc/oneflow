#ifndef ONEFLOW_CORE_JOB_SBP_INFER_HINT_H_
#define ONEFLOW_CORE_JOB_SBP_INFER_HINT_H_

#include "oneflow/core/job/sbp_infer_hint_conf.pb.h"

namespace oneflow {

class SbpInferHint final {
 public:
  SbpInferHint(bool is_model_blob, int64_t parallel_num, int64_t num_axes, int64_t split_axis)
      : is_model_blob_(is_model_blob),
        parallel_num_(parallel_num),
        num_axes_(num_axes),
        split_axis_(split_axis) {}
  SbpInferHint(const SbpInferHint&) = default;
  ~SbpInferHint() = default;

  // Getters
  bool is_model_blob() const { return is_model_blob_; }
  int64_t parallel_num() const;
  int64_t num_axes() const;
  int64_t split_axis() const;
  bool has_split_axis() const { return split_axis_ == -1; }
  bool is_model_split() const;
  bool is_model_broadcast() const;
  bool is_data_split() const;
  bool is_data_partial_sum() const;
  bool is_data_blob() const { return !is_model_blob(); }

 private:
  const bool is_model_blob_;
  const int64_t parallel_num_;
  const int64_t num_axes_;
  const int64_t split_axis_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SBP_INFER_HINT_H_
