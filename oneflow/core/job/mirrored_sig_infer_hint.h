#ifndef ONEFLOW_CORE_JOB_MIRRORED_SIG_INFER_HINT_H_
#define ONEFLOW_CORE_JOB_MIRRORED_SIG_INFER_HINT_H_

#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

class MirroredSigInferHint final {
 public:
  MirroredSigInferHint(const ParallelDesc* parallel_desc, bool is_mirrored_parallel_view)
      : parallel_desc_(parallel_desc), is_mirrored_parallel_view_(is_mirrored_parallel_view) {}

  const ParallelDesc& parallel_desc() const { return *parallel_desc_; }
  bool is_mirrored_parallel_view() const { return is_mirrored_parallel_view_; }

 private:
  const ParallelDesc* parallel_desc_;
  bool is_mirrored_parallel_view_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_MIRRORED_SIG_INFER_HINT_H_
