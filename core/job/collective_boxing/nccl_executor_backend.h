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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_NCCL_EXECUTOR_BACKEND_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_NCCL_EXECUTOR_BACKEND_H_

#include "oneflow/core/job/collective_boxing/executor_backend.h"

#ifdef WITH_CUDA

namespace oneflow {

namespace boxing {

namespace collective {

struct RequestId;

class NcclExecutorBackend : public ExecutorBackend {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclExecutorBackend);
  NcclExecutorBackend();
  ~NcclExecutorBackend() override;

 private:
  void Init(std::shared_ptr<RequestStore> request_store) override;
  void InitJob(int64_t job_id) override;
  void DeinitJob(int64_t job_id) override;
  void GroupRequests(const std::vector<RequestId>& request_ids,
                     const std::function<void(std::vector<RequestId>&&, void*)>& Handler) override;
  void ExecuteGroup(void* group_token) override;
  void* CreateGroupToken(const std::vector<RequestId>& group) override;
  void DestroyGroupToken(void* group_token) override;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_NCCL_EXECUTOR_BACKEND_H_
