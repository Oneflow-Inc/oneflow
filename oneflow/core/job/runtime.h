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
#ifndef ONEFLOW_CORE_JOB_RUNTIME_H_
#define ONEFLOW_CORE_JOB_RUNTIME_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

class Runtime final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Runtime);
  Runtime() = delete;
  ~Runtime();

  // TODO(chengcheng): refactor Runtime interface about variable_op_name2eager_blob
  Runtime(const Plan& plan, const HashMap<std::string, Blob*>& variable_op_name2eager_blob);

 private:
  void NewAllGlobal(const Plan& plan,
                    const HashMap<std::string, Blob*>& variable_op_name2eager_blob);
  void DeleteAllGlobal();

  HashMap<int64_t, int64_t> job_id2actor_size_;
};

}  // namespace oneflow

#endif
