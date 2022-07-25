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
#ifndef ONEFLOW_CORE_JOB_OF_COLLECTIVE_BOXING_COLLECTIVE_BACKEND_OFCCL_H_
#define ONEFLOW_CORE_JOB_OF_COLLECTIVE_BOXING_COLLECTIVE_BACKEND_OFCCL_H_

#include "oneflow/core/job/of_collective_boxing/collective_backend.h"

#ifdef WITH_CUDA

namespace oneflow {

namespace boxing {

namespace of_collective {

struct OfRequestId;

class CollectiveBackendOfccl : public CollectiveBackend {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBackendOfccl);
  CollectiveBackendOfccl();
  ~CollectiveBackendOfccl() override;

 private:
  void Init(std::shared_ptr<OfRequestStore> request_store) override;
  void InitJob(int64_t job_id) override;
  void DeinitJob(int64_t job_id) override;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace of_collective

}  // namespace boxing

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_JOB_OF_COLLECTIVE_BOXING_COLLECTIVE_BACKEND_OFCCL_H_
