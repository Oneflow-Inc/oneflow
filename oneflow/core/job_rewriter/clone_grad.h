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
#ifndef ONEFLOW_CORE_JOB_REWRITER_CLONE_GRAD_H_
#define ONEFLOW_CORE_JOB_REWRITER_CLONE_GRAD_H_

#include "oneflow/core/job_rewriter/autograd.h"

namespace oneflow {

Maybe<void> GenerateCloneGradOpIfNeed(
    const OpNode& op_node, JobBuilder* job_builder,
    const HashMap<OpBlobArg, LogicalBlobId>& in_oba2in_diff_lbi,
    HashMap<OpBlobArg, LogicalBlobId>* out_oba2out_diff_lbi,
    HashMap<OpBlobArg, LogicalBlobId>* out_oba2clone_bw_add_out_lbi);
}

#endif  // ONEFLOW_CORE_JOB_REWRITER_CLONE_GRAD_H_
