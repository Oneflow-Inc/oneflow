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
#ifndef ONEFLOW_CORE_JOB_REWRITER_OP_COLLECTION_JOB_PASS_STATE_H_
#define ONEFLOW_CORE_JOB_REWRITER_OP_COLLECTION_JOB_PASS_STATE_H_

#include "oneflow/core/job_rewriter/job_pass.h"

namespace oneflow {

class OpCollectionJobPassState : public JobPassState {
 public:
  explicit OpCollectionJobPassState(const std::string& op_collection)
      : op_collection_(op_collection) {}

 private:
  HashMap<int64_t, int64_t> scope_id2current_op_collection_scope_id_;
  std::string op_collection_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_OP_COLLECTION_JOB_PASS_STATE_H_
