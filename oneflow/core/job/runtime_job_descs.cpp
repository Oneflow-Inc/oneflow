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
#include "oneflow/core/job/runtime_job_descs.h"

namespace oneflow {

RuntimeJobDescs::RuntimeJobDescs(const PbMap<int64_t, JobConfigProto>& proto) {
  for (const auto& pair : proto) {
    auto job_desc = std::make_unique<JobDesc>(pair.second, pair.first);
    CHECK(job_id2job_desc_.emplace(pair.first, std::move(job_desc)).second);
  }
}

}  // namespace oneflow
