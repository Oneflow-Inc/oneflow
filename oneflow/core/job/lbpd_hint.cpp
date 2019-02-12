#include "oneflow/core/job/lbpd_hint.h"
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

const SplitParallel& SbpInferHint::model_split() const {
  CHECK(is_model_blob());
  return sbp_infer_hint_conf_.sbp_parallel().split_parallel();
}
const BroadcastParallel& SbpInferHint::model_clone() const {
  CHECK(is_model_blob());
  return sbp_infer_hint_conf_.sbp_parallel().broadcast_parallel();
}
const SplitParallel& SbpInferHint::data_split() const {
  CHECK(is_data_blob());
  return sbp_infer_hint_conf_.sbp_parallel().split_parallel();
}
const PartialSumParallel& SbpInferHint::data_partial_sum() const {
  CHECK(is_data_blob());
  return sbp_infer_hint_conf_.sbp_parallel().partial_sum_parallel();
}
bool SbpInferHint::has_model_split() const {
  return is_model_blob() && sbp_infer_hint_conf_.sbp_parallel().has_split_parallel();
}
bool SbpInferHint::has_model_clone() const {
  return is_model_blob() && sbp_infer_hint_conf_.sbp_parallel().has_broadcast_parallel();
}
bool SbpInferHint::has_data_split() const {
  return is_data_blob() && sbp_infer_hint_conf_.sbp_parallel().has_split_parallel();
}
bool SbpInferHint::has_data_partial_sum() const {
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

bool SbpInferHint::operator==(const SbpInferHint& rhs) const {
  return parallel_num_ == rhs.parallel_num_ && num_axes_ == rhs.num_axes_
         && PbMd().Equivalent(sbp_infer_hint_conf_, rhs.sbp_infer_hint_conf_);
}

}  // namespace oneflow
