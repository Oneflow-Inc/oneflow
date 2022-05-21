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
#include "oneflow/core/job/collective_boxing/collective_manager.h"
#include "oneflow/core/job/collective_boxing/collective_builder.h"
#include "oneflow/core/job/collective_boxing/request_store.h"
#include "oneflow/core/job/collective_boxing/coordinator.h"
#include "oneflow/core/job/collective_boxing/static_group_coordinator.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/collective_boxing/nccl_executor_backend.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace boxing {

namespace collective {

class CollectiveBuilderImpl : public CollectiveBuilder {
 public:
  CollectiveBuilderImpl() = default;
  ~CollectiveBuilderImpl() override = default;

  void Init(std::shared_ptr<RequestStore> request_store) override;
  void InitJob(int64_t job_id) override;
  void DeinitJob(int64_t job_id) override;

 private:
  std::vector<std::unique_ptr<ExecutorBackend>> backends_;
  std::shared_ptr<RequestStore> request_store_;
  std::vector<RequestId> group_buffer_;
};

void CollectiveBuilderImpl::Init(std::shared_ptr<RequestStore> request_store) {
  request_store_ = request_store;
  backends_.resize(Backend_ARRAYSIZE);
#ifdef WITH_CUDA
  int cuda_dev_count = 0;
  cudaError_t err = cudaGetDeviceCount(&cuda_dev_count);
  if (err != cudaErrorNoDevice && err != cudaErrorInsufficientDriver) { OF_CUDA_CHECK(err); }
  if (cuda_dev_count > 0) {
    std::unique_ptr<ExecutorBackend> nccl_backend = std::make_unique<NcclExecutorBackend>();
    nccl_backend->Init(request_store_);
    // TODO: (Panlichen) define Backend::kBackendOFCCL to get rid of ExecutorBackend and NcclExecutorBackend
    backends_.at(Backend::kBackendNCCL) = std::move(nccl_backend);
  }
#endif
}

void CollectiveBuilderImpl::InitJob(int64_t job_id) {
#ifdef WITH_CUDA
  if (backends_.at(Backend::kBackendNCCL)) { backends_.at(Backend::kBackendNCCL)->InitJob(job_id); }
#endif
}

void CollectiveBuilderImpl::DeinitJob(int64_t job_id) {
#ifdef WITH_CUDA
  if (backends_.at(Backend::kBackendNCCL)) {
    backends_.at(Backend::kBackendNCCL)->DeinitJob(job_id);
  }
#endif
}

struct CollectiveMgr::Impl {
  Impl();
  std::shared_ptr<RequestStore> request_store;
  std::shared_ptr<CollectiveBuilder> collective_builder;
};

CollectiveMgr::Impl::Impl() {
  request_store.reset(new RequestStore());
  collective_builder.reset(new CollectiveBuilderImpl());
  collective_builder->Init(request_store);
}

class CollectiveMgrPlanToken {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveMgrPlanToken);
  explicit CollectiveMgrPlanToken(const std::vector<int64_t>& job_ids) : job_ids_(job_ids) {}
  ~CollectiveMgrPlanToken() = default;
  const std::vector<int64_t>& job_ids() const {return job_ids_; }

 private:
  std::vector<int64_t> job_ids_;
};

CollectiveMgrPlanToken* CollectiveMgr::AddPlan(const Plan& plan) {
  std::vector<int64_t> job_ids;
  // resue plan.collective_boxing_plan()
  for (const auto& job_id7request_set : plan.collective_boxing_plan().job_id2request_set()) {
    const int64_t job_id = job_id7request_set.first;
    job_ids.emplace_back(job_id);
    impl_->request_store->InitJob(job_id, job_id7request_set.second);
    impl_->collective_builder->InitJob(job_id);
  }
  return new CollectiveMgrPlanToken(job_ids);
}

void CollectiveMgr::DeletePlan(CollectiveMgrPlanToken* plan_token) {
  const std::vector<int64_t>& job_ids = plan_token->job_ids();
  for (const auto& job_id : job_ids) {
    impl_->collective_builder->DeinitJob(job_id);
    impl_->request_store->DeinitJob(job_id);
  }
  delete plan_token;
}

CollectiveMgr::CollectiveMgr() { impl_.reset(new Impl()); }

CollectiveMgr::~CollectiveMgr() = default;

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
