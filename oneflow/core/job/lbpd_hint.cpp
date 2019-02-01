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
  CHECK(lbpd_hint_conf_.has_model_split());
  return lbpd_hint_conf_.model_split();
}
const CloneParallel& LbpdHint::model_clone() const {
  CHECK(lbpd_hint_conf_.has_model_clone());
  return lbpd_hint_conf_.model_clone();
}
const SplitParallel& LbpdHint::data_split() const {
  CHECK(lbpd_hint_conf_.has_data_split());
  return lbpd_hint_conf_.data_split();
}
const PartialSumParallel& LbpdHint::data_partial_sum() const {
  CHECK(lbpd_hint_conf_.has_data_partial_sum());
  return lbpd_hint_conf_.data_partial_sum();
}

bool LbpdHint::operator==(const LbpdHint& rhs) const {
  return parallel_num_ == rhs.parallel_num_ && num_axes_ == rhs.num_axes_
         && PbMd().Equivalent(lbpd_hint_conf_, rhs.lbpd_hint_conf_);
}

}  // namespace oneflow
