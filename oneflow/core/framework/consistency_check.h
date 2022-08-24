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
#ifndef ONEFLOW_CORE_FRAMEWORK_DATA_CONSISTENCY_CHECK_H_
#define ONEFLOW_CORE_FRAMEWORK_DATA_CONSISTENCY_CHECK_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/common/tensor_meta.h"

namespace oneflow {

class NonRecursiveMetaInfoConsistencyCheckScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonRecursiveMetaInfoConsistencyCheckScope);
  NonRecursiveMetaInfoConsistencyCheckScope();
  ~NonRecursiveMetaInfoConsistencyCheckScope();
};

Maybe<void> DataConsistencyCheck(const void* buffer_ptr, size_t buffer_size,
                                 Symbol<ParallelDesc> placement);

Maybe<void> MetaInfoConsistencyCheck(const Symbol<ParallelDesc>& placement,
                                     const Optional<Symbol<NdSbp>>& nd_sbp,
                                     const Optional<Symbol<NdSbp>>& grad_nd_sbp,
                                     const size_t debug_level, bool force_check);

Maybe<void> MetaInfoConsistencyCheck(const Symbol<ParallelDesc>& placement,
                                     const Optional<Symbol<NdSbp>>& nd_sbp,
                                     const size_t debug_level, bool force_check);

Maybe<void> MetaInfoConsistencyCheck(const Symbol<ParallelDesc>& placement,
                                     const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                                     const std::vector<Symbol<SbpParallel>>& grad_sbp_tuple,
                                     const size_t debug_level, bool force_check);

Maybe<void> MetaInfoConsistencyCheck(const Symbol<ParallelDesc>& placement,
                                     const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                                     const size_t debug_level, bool force_check);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_DATA_CONSISTENCY_CHECK_H_
