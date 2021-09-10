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

#ifdef WITH_CUDA

#include "oneflow/core/job/collective_boxing/executor_backend.h"

namespace oneflow {

namespace boxing {

namespace collective {

class NcclExecutorBackend : public ExecutorBackend {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclExecutorBackend);
  NcclExecutorBackend();
  ~NcclExecutorBackend() override;

 private:
  void Init(std::shared_ptr<RequestStore> request_store) override;
  void AddPlan(const std::vector<int64_t>& job_ids) override;
  void DeletePlan(const std::vector<int64_t>& job_ids) override;
  void GroupRequests(int64_t job_id, const std::vector<int32_t>& request_ids,
                     const std::function<void(int64_t, std::vector<int32_t>&&)>& Handler) override;
  void ExecuteRequests(int64_t job_id, const std::vector<int32_t>& request_ids,
                       void* executor_token) override;
  void* CreateExecutorToken(int64_t job_id, int32_t request_id) override;
  void DestroyExecutorToken(void* executor_token) override;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_NCCL_EXECUTOR_BACKEND_H_
