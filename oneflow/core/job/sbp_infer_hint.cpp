#include "oneflow/core/job/sbp_infer_hint.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

bool SbpInferHint::has_split_axis() const {
  bool ret = sbp_parallel_.has_split_parallel();
  if (ret) {
    CHECK_GE(sbp_parallel_.split_parallel().axis(), 0);
    CHECK_LT(sbp_parallel_.split_parallel().axis(), num_axes());
  }
  return ret;
}

int64_t SbpInferHint::split_axis() const {
  CHECK(has_split_axis());
  return sbp_parallel_.split_parallel().axis();
}

bool SbpInferHint::is_model_split() const { return is_model_blob() && has_split_axis(); }
bool SbpInferHint::is_model_broadcast() const {
  return is_model_blob() && sbp_parallel_.has_broadcast_parallel();
}
bool SbpInferHint::is_data_split() const { return is_data_blob() && has_split_axis(); }
bool SbpInferHint::is_data_partial_sum() const {
  return is_data_blob() && sbp_parallel_.has_partial_sum_parallel();
}

}  // namespace oneflow
