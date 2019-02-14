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

bool SbpInferHint::has_split_axis() const { return split_axis_ >= 0 && split_axis_ < num_axes(); }

int64_t SbpInferHint::split_axis() const {
  CHECK(has_split_axis());
  return split_axis_;
}

bool SbpInferHint::is_model_split() const { return is_model_blob() && has_split_axis(); }
bool SbpInferHint::is_model_broadcast() const { return is_model_blob() && !has_split_axis(); }
bool SbpInferHint::is_data_split() const { return is_data_blob() && has_split_axis(); }
bool SbpInferHint::is_data_partial_sum() const { return is_data_blob() && !has_split_axis(); }

}  // namespace oneflow
