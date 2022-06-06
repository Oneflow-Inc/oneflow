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
#include "oneflow/core/job/of_collective_boxing/collective_manager.h"
#include "oneflow/core/job/of_collective_boxing/collective_builder.h"
#include "oneflow/core/job/of_collective_boxing/of_request_store.h"
#include "oneflow/core/graph/boxing/of_collective_boxing_util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/of_collective_boxing/collective_backend_ofccl.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace boxing {

namespace of_collective {

class CollectiveBuilderImpl : public CollectiveBuilder {
 public:
  CollectiveBuilderImpl() = default;
  ~CollectiveBuilderImpl() override = default;

  void Init(std::shared_ptr<OfRequestStore> request_store) override;
  void InitJob(int64_t job_id) override;
  void DeinitJob(int64_t job_id) override;

 private:
  std::vector<std::unique_ptr<CollectiveBackend>> backends_;
  std::shared_ptr<OfRequestStore> request_store_;
};

void CollectiveBuilderImpl::Init(std::shared_ptr<OfRequestStore> request_store) {
  request_store_ = request_store;
  backends_.resize(Backend_ARRAYSIZE);
#ifdef WITH_CUDA
  int cuda_dev_count = 0;
  cudaError_t err = cudaGetDeviceCount(&cuda_dev_count);
  if (err != cudaErrorNoDevice && err != cudaErrorInsufficientDriver) { OF_CUDA_CHECK(err); }
  if (cuda_dev_count > 0) {
    std::unique_ptr<CollectiveBackend> ofccl_backend = std::make_unique<CollectiveBackendOfccl>();
    ofccl_backend->Init(request_store_);
    backends_.at(Backend::kBackendOFCCL) = std::move(ofccl_backend);
  }
#endif
}

void CollectiveBuilderImpl::InitJob(int64_t job_id) {
#ifdef WITH_CUDA
  if (backends_.at(Backend::kBackendOFCCL)) { backends_.at(Backend::kBackendOFCCL)->InitJob(job_id); }
#endif
}

void CollectiveBuilderImpl::DeinitJob(int64_t job_id) {
#ifdef WITH_CUDA
  if (backends_.at(Backend::kBackendOFCCL)) {
    backends_.at(Backend::kBackendOFCCL)->DeinitJob(job_id);
  }
#endif
}

struct CollectiveMgr::Impl {
  Impl();
  std::shared_ptr<OfRequestStore> request_store;
  std::shared_ptr<CollectiveBuilder> collective_builder;
};

CollectiveMgr::Impl::Impl() {
  request_store.reset(new OfRequestStore());
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
  for (const auto& job_id7request_set : plan.of_collective_boxing_plan().job_id2request_set()) {
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

}  // namespace of_collective

}  // namespace boxing

}  // namespace oneflow
