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
#ifndef ONEFLOW_CORE_JOB_PARALLEL_DISTRIBUTION_INFER_HINT_H_
#define ONEFLOW_CORE_JOB_PARALLEL_DISTRIBUTION_INFER_HINT_H_

#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

class ParallelDistributionInferHint final {
 public:
  ParallelDistributionInferHint(const ParallelDesc* parallel_desc,
                                const BlobDesc* logical_blob_desc,
                                const ParallelDistribution* parallel_distribution)
      : parallel_desc_(parallel_desc),
        logical_blob_desc_(logical_blob_desc),
        parallel_distribution_(parallel_distribution) {}
  ParallelDistributionInferHint(const ParallelDistributionInferHint&) = default;
  ~ParallelDistributionInferHint() = default;

  // Getters
  const ParallelDesc& parallel_desc() const { return *parallel_desc_; }
  const BlobDesc& logical_blob_desc() const { return *logical_blob_desc_; }
  const ParallelDistribution& parallel_distribution() const { return *parallel_distribution_; }

 private:
  const ParallelDesc* parallel_desc_;
  const BlobDesc* logical_blob_desc_;
  const ParallelDistribution* parallel_distribution_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_PARALLEL_DISTRIBUTION_INFER_HINT_H_
