#ifndef ONEFLOW_CORE_JOB_SBP_INFER_HINT_H_
#define ONEFLOW_CORE_JOB_SBP_INFER_HINT_H_

#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

class SbpInferHint final {
 public:
  SbpInferHint(const ParallelDesc* parallel_desc, const BlobDesc* logical_blob_desc,
               const SbpParallel* sbp_parallel, const OptInt64* batch_axis)
      : parallel_desc_(parallel_desc),
        logical_blob_desc_(logical_blob_desc),
        sbp_parallel_(sbp_parallel),
        batch_axis_(batch_axis) {}
  SbpInferHint(const SbpInferHint&) = default;
  ~SbpInferHint() = default;

  // Getters
  const ParallelDesc& parallel_desc() const { return *parallel_desc_; }
  const BlobDesc& logical_blob_desc() const { return *logical_blob_desc_; }
  const SbpParallel& sbp_parallel() const { return *sbp_parallel_; }
  const OptInt64& batch_axis() const { return *batch_axis_; }

 private:
  const ParallelDesc* parallel_desc_;
  const BlobDesc* logical_blob_desc_;
  const SbpParallel* sbp_parallel_;
  const OptInt64* batch_axis_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SBP_INFER_HINT_H_
