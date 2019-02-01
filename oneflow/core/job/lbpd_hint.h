#ifndef ONEFLOW_CORE_JOB_LBPD_HINT_H_
#define ONEFLOW_CORE_JOB_LBPD_HINT_H_

#include "oneflow/core/job/lbpd_hint_conf.pb.h"

namespace oneflow {

class LbpdHint final {
 public:
  LbpdHint() : parallel_num_(-1), num_axes_(-1) {}
  LbpdHint(const LbpdHint&) = default;
  ~LbpdHint() = default;

  // Getters
  int64_t parallel_num() const;
  int64_t num_axes() const;
  const SplitParallel& model_split() const;
  const CloneParallel& model_clone() const;
  const SplitParallel& data_split() const;
  const PartialSumParallel& data_partial_sum() const;
  bool has_model_split() const { return lbpd_hint_conf_.has_model_split(); }
  bool has_model_clone() const { return lbpd_hint_conf_.has_model_clone(); }
  bool has_data_split() const { return lbpd_hint_conf_.has_data_split(); }
  bool has_data_partial_sum() const { return lbpd_hint_conf_.has_data_partial_sum(); }
  bool is_model_blob() const { return has_model_split() || has_model_clone(); }
  bool is_data_blob() const { return has_data_split() || has_data_partial_sum(); }

  // Setters
  void set_parallel_num(int64_t val) { parallel_num_ = val; }
  void set_num_axes(int64_t val) { num_axes_ = val; }
  SplitParallel* mutable_model_split() { return lbpd_hint_conf_.mutable_model_split(); }
  CloneParallel* mutable_model_clone() { return lbpd_hint_conf_.mutable_model_clone(); }
  SplitParallel* mutable_data_split() { return lbpd_hint_conf_.mutable_data_split(); }
  PartialSumParallel* mutable_data_partial_sum() {
    return lbpd_hint_conf_.mutable_data_partial_sum();
  }

  bool operator==(const LbpdHint& rhs) const;
  bool operator!=(const LbpdHint& rhs) const { return !(*this == rhs); }

 private:
  int64_t parallel_num_;
  int64_t num_axes_;
  LbpdHintConf lbpd_hint_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_LBPD_HINT_H_
