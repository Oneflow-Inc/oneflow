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
#ifndef ONEFLOW_CORE_JOB_JOB_INSTANCE_H_
#define ONEFLOW_CORE_JOB_JOB_INSTANCE_H_

#include "oneflow/core/register/ofblob.h"

namespace oneflow {

class ForeignJobInstance {
 public:
  ForeignJobInstance() = default;

  virtual ~ForeignJobInstance() = default;

  virtual std::string job_name() const { UNIMPLEMENTED(); }
  virtual std::string sole_input_op_name_in_user_job() const { UNIMPLEMENTED(); }
  virtual std::string sole_output_op_name_in_user_job() const { UNIMPLEMENTED(); }
  virtual void PushBlob(uint64_t ofblob_ptr) const { UNIMPLEMENTED(); }
  virtual void PullBlob(uint64_t ofblob_ptr) const { UNIMPLEMENTED(); }
  virtual void Finish() const { UNIMPLEMENTED(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_INSTANCE_H_
