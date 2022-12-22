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
#ifndef ONEFLOW_CORE_JOB_JOB_IR_H_
#define ONEFLOW_CORE_JOB_JOB_IR_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

Maybe<std::string> ConvertJobToTosaIR(Job* job);
Maybe<std::string> ConvertJobToIR(Job* job);
Maybe<void> SaveJobToIR(Job* job, const std::string& path);
Maybe<void> LoadJobFromIR(Job* job, const std::string& path);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_IR_H_
