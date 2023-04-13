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
#include "oneflow/core/job/runtime_buffer_managers_scope.h"
#include "oneflow/core/job/job_instance.h"

namespace oneflow {

RuntimeBufferManagersScope::RuntimeBufferManagersScope() {
  Singleton<BufferMgr<int64_t>>::New();
  Singleton<BufferMgr<std::shared_ptr<JobInstance>>>::New();
}

RuntimeBufferManagersScope::~RuntimeBufferManagersScope() {
  Singleton<BufferMgr<std::shared_ptr<JobInstance>>>::Delete();
  Singleton<BufferMgr<int64_t>>::Delete();
}

}  // namespace oneflow
