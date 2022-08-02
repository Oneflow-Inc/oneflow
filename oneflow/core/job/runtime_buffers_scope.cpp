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
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/runtime_buffers_scope.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_instance.h"

namespace oneflow {

RuntimeBuffersScope::RuntimeBuffersScope(const JobConfs& job_confs) {
  size_t job_size = Singleton<JobName2JobId>::Get()->size();
  Singleton<BufferMgr<int64_t>>::Get()->NewBuffer(kBufferNameGlobalWaitJobId, job_size);
  auto* buffer_mgr = Singleton<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
  for (const auto& pair : job_confs.job_id2job_conf()) {
    const auto& job_name = pair.second.job_name();
    CHECK_EQ(pair.first, Singleton<JobName2JobId>::Get()->at(job_name));
    size_t concurrency_width = pair.second.concurrency_width();
    buffer_mgr->NewBuffer(GetCallbackNotifierBufferName(job_name), concurrency_width);
  }
}

RuntimeBuffersScope::~RuntimeBuffersScope() {
  auto* buffer_mgr = Singleton<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
  for (const auto& pair : *Singleton<JobName2JobId>::Get()) {
    const auto& job_name = pair.first;
    buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Close();
  }
  Singleton<BufferMgr<int64_t>>::Get()->Get(kBufferNameGlobalWaitJobId)->Close();
}

}  // namespace oneflow
