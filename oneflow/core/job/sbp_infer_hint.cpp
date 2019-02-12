#include "oneflow/core/job/sbp_infer_hint.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

int64_t SbpInferHint::parallel_num() const {
  CHECK_GT(parallel_num_, 0);
  return parallel_num_;
}

int64_t SbpInferHint::num_axes() const {
  CHECK_GT(num_axes_, 0);
  return num_axes_;
}

int64_t SbpInferHint::split_axis() const {
  CHECK_GT(split_axis_, 0);
  return split_axis_;
}

bool SbpInferHint::is_model_split() const {
  return is_model_blob() && sbp_infer_hint_conf_.sbp_parallel().has_split_parallel();
}
bool SbpInferHint::is_model_broadcast() const {
  return is_model_blob() && sbp_infer_hint_conf_.sbp_parallel().has_broadcast_parallel();
}
bool SbpInferHint::is_data_split() const {
  return is_data_blob() && sbp_infer_hint_conf_.sbp_parallel().has_split_parallel();
}
bool SbpInferHint::is_data_partial_sum() const {
  return is_data_blob() && sbp_infer_hint_conf_.sbp_parallel().has_partial_sum_parallel();
}

SplitParallel* SbpInferHint::mutable_model_split() {
  sbp_infer_hint_conf_.set_is_model_blob(true);
  return sbp_infer_hint_conf_.mutable_sbp_parallel()->mutable_split_parallel();
}
BroadcastParallel* SbpInferHint::mutable_model_clone() {
  sbp_infer_hint_conf_.set_is_model_blob(true);
  return sbp_infer_hint_conf_.mutable_sbp_parallel()->mutable_broadcast_parallel();
}
SplitParallel* SbpInferHint::mutable_data_split() {
  sbp_infer_hint_conf_.set_is_model_blob(false);
  return sbp_infer_hint_conf_.mutable_sbp_parallel()->mutable_split_parallel();
}
PartialSumParallel* SbpInferHint::mutable_data_partial_sum() {
  sbp_infer_hint_conf_.set_is_model_blob(false);
  return sbp_infer_hint_conf_.mutable_sbp_parallel()->mutable_partial_sum_parallel();
}

}  // namespace oneflow
