#include "oneflow/core/job/lbpd_hint.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

int64_t LbpdHint::parallel_num() const {
  CHECK_GT(parallel_num_, 0);
  return parallel_num_;
}

int64_t LbpdHint::num_axes() const {
  CHECK_GT(num_axes_, 0);
  return num_axes_;
}

const SplitParallel& LbpdHint::model_split() const {
  CHECK(is_model_blob());
  return lbpd_hint_conf_.sbp_parallel().split();
}
const BroadcastParallel& LbpdHint::model_clone() const {
  CHECK(is_model_blob());
  return lbpd_hint_conf_.sbp_parallel().broadcast();
}
const SplitParallel& LbpdHint::data_split() const {
  CHECK(is_data_blob());
  return lbpd_hint_conf_.sbp_parallel().split();
}
const PartialSumParallel& LbpdHint::data_partial_sum() const {
  CHECK(is_data_blob());
  return lbpd_hint_conf_.sbp_parallel().partial_sum();
}
bool LbpdHint::has_model_split() const {
  return is_model_blob() && lbpd_hint_conf_.sbp_parallel().has_split();
}
bool LbpdHint::has_model_clone() const {
  return is_model_blob() && lbpd_hint_conf_.sbp_parallel().has_broadcast();
}
bool LbpdHint::has_data_split() const {
  return is_data_blob() && lbpd_hint_conf_.sbp_parallel().has_split();
}
bool LbpdHint::has_data_partial_sum() const {
  return is_data_blob() && lbpd_hint_conf_.sbp_parallel().has_partial_sum();
}

SplitParallel* LbpdHint::mutable_model_split() {
  lbpd_hint_conf_.set_is_model_blob(true);
  return lbpd_hint_conf_.mutable_sbp_parallel()->mutable_split();
}
BroadcastParallel* LbpdHint::mutable_model_clone() {
  lbpd_hint_conf_.set_is_model_blob(true);
  return lbpd_hint_conf_.mutable_sbp_parallel()->mutable_broadcast();
}
SplitParallel* LbpdHint::mutable_data_split() {
  lbpd_hint_conf_.set_is_model_blob(false);
  return lbpd_hint_conf_.mutable_sbp_parallel()->mutable_split();
}
PartialSumParallel* LbpdHint::mutable_data_partial_sum() {
  lbpd_hint_conf_.set_is_model_blob(false);
  return lbpd_hint_conf_.mutable_sbp_parallel()->mutable_partial_sum();
}

bool LbpdHint::operator==(const LbpdHint& rhs) const {
  return parallel_num_ == rhs.parallel_num_ && num_axes_ == rhs.num_axes_
         && PbMd().Equivalent(lbpd_hint_conf_, rhs.lbpd_hint_conf_);
}

}  // namespace oneflow
