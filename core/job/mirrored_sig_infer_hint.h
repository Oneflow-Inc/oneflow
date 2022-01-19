/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
