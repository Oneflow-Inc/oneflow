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
#include "oneflow/core/job/collective_boxing/scheduler.h"
#include "oneflow/core/job/collective_boxing/executor.h"
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

namespace {

bool CanMergeIntoCurGroup(RequestStore* request_store, const RequestEntry* request_entry,
                          const RequestId& request_id, const std::vector<RequestId>& group_buffer) {
  if (group_buffer.empty()) { return true; }
  const RequestId& group_entry_id = group_buffer.front();
  const auto* group_entry = request_store->MutRequestEntry(group_entry_id);
  return (request_id.job_id == group_entry_id.job_id
          && request_entry->desc().dependency_depth() == group_entry->desc().dependency_depth()
          && request_entry->desc().op_desc().backend() == group_entry->desc().op_desc().backend()
          && request_entry->device_set_symbol() == group_entry->device_set_symbol());
}

bool HasRankInteraction(const DeviceSet& a, const DeviceSet& b) {
  for (int64_t i = 0; i < a.device_size(); ++i) {
    const DeviceDesc& a_device_desc = a.device(i);
    for (int64_t j = 0; j < b.device_size(); ++j) {
      if (a_device_desc.machine_id() == b.device(j).machine_id()) { return true; }
    }
  }
  return false;
}

}  // namespace

class RequestHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RequestHandle);
  RequestHandle(int32_t local_rank, void* request_entry_token, void* coordinator_token)
      : local_rank_(local_rank),
        request_entry_token_(request_entry_token),
        coordinator_token_(coordinator_token) {}
  ~RequestHandle() = default;

  int32_t local_rank() const { return local_rank_; }

  void* request_entry_token() { return request_entry_token_; }

  void* coordinator_token() { return coordinator_token_; }

 private:
  int32_t local_rank_;
  void* request_entry_token_;
  void* coordinator_token_;
};

class GroupToken final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GroupToken);
  GroupToken(Backend backend, void* backend_group_token)
      : backend_(backend), backend_group_token_(backend_group_token) {}
  ~GroupToken() = default;

  Backend backend() { return backend_; }

  void* backend_group_token() { return backend_group_token_; }

 private:
  Backend backend_;
  void* backend_group_token_;
};

class ExecutorImpl : public Executor {
 public:
  ExecutorImpl() = default;
  ~ExecutorImpl() override = default;

  void Init(std::shared_ptr<RequestStore> request_store) override;
  void InitJob(int64_t job_id) override;
  void DeinitJob(int64_t job_id) override;
  void GroupRequests(
      const std::vector<RequestId>& request_ids,
      const std::function<void(std::vector<RequestId>&&, GroupToken*)>& Handler) override;
  void ExecuteGroup(GroupToken* group_token) override;
  void DestroyGroupToken(GroupToken* group_token) override;

 private:
  Backend GetUniqueBackend(const std::vector<RequestId>& group);
  GroupToken* CreateGroupToken(const std::vector<RequestId>& group, void* backend_group_token);

  std::vector<std::unique_ptr<ExecutorBackend>> backends_;
  std::shared_ptr<RequestStore> request_store_;
  std::vector<RequestId> group_buffer_;
};

void ExecutorImpl::Init(std::shared_ptr<RequestStore> request_store) {
  request_store_ = request_store;
  backends_.resize(Backend_ARRAYSIZE);
#ifdef WITH_CUDA
  int cuda_dev_count = 0;
  cudaError_t err = cudaGetDeviceCount(&cuda_dev_count);
  if (err != cudaErrorNoDevice && err != cudaErrorInsufficientDriver) { OF_CUDA_CHECK(err); }
  if (cuda_dev_count > 0) {
    std::unique_ptr<ExecutorBackend> nccl_backend = std::make_unique<NcclExecutorBackend>();
    nccl_backend->Init(request_store_);
    backends_.at(Backend::kBackendNCCL) = std::move(nccl_backend);
  }
#endif
}

void ExecutorImpl::InitJob(int64_t job_id) {
#ifdef WITH_CUDA
  if (backends_.at(Backend::kBackendNCCL)) { backends_.at(Backend::kBackendNCCL)->InitJob(job_id); }
#endif
}

void ExecutorImpl::DeinitJob(int64_t job_id) {
#ifdef WITH_CUDA
  if (backends_.at(Backend::kBackendNCCL)) {
    backends_.at(Backend::kBackendNCCL)->DeinitJob(job_id);
  }
#endif
}

GroupToken* ExecutorImpl::CreateGroupToken(const std::vector<RequestId>& group,
                                           void* backend_group_token) {
  return new GroupToken(GetUniqueBackend(group), backend_group_token);
}

void ExecutorImpl::DestroyGroupToken(GroupToken* group_token) {
  backends_.at(Backend::kBackendNCCL)->DestroyGroupToken(group_token->backend_group_token());
  delete group_token;
}

void ExecutorImpl::GroupRequests(
    const std::vector<RequestId>& request_ids,
    const std::function<void(std::vector<RequestId>&&, GroupToken*)>& Handler) {
  if (request_ids.empty()) { return; }
  const CollectiveBoxingConf& conf =
      Singleton<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  auto BackendHandler = [&](std::vector<RequestId>&& group, void* backend_group_token) {
    GroupToken* group_token = CreateGroupToken(group, backend_group_token);
    Handler(std::move(group), group_token);
  };
  auto HandleGroup = [&]() {
    if (group_buffer_.empty()) { return; }
    const auto backend =
        request_store_->MutRequestEntry(group_buffer_.front())->desc().op_desc().backend();
    backends_.at(backend)->GroupRequests(group_buffer_, BackendHandler);
    group_buffer_.clear();
  };
  request_store_->ForEachMutRequestEntryForIdsInJob(
      request_ids, [&](RequestEntry* request_entry, int32_t i, const RequestId& request_id) {
        if (request_entry->HasRankOnThisNode()) {
          if (!(conf.enable_fusion()
                && CanMergeIntoCurGroup(request_store_.get(), request_entry, request_id,
                                        group_buffer_))) {
            HandleGroup();
          }
          group_buffer_.emplace_back(request_id);
        } else {
          if (!group_buffer_.empty()
              && HasRankInteraction(
                  request_store_->MutRequestEntry(group_buffer_.back())->desc().device_set(),
                  request_entry->desc().device_set())) {
            HandleGroup();
          }
        }
      });
  HandleGroup();
}

void ExecutorImpl::ExecuteGroup(GroupToken* group_token) {
  const Backend backend = group_token->backend();
  backends_.at(backend)->ExecuteGroup(group_token->backend_group_token());
}

Backend ExecutorImpl::GetUniqueBackend(const std::vector<RequestId>& group) {
  const Backend backend =
      request_store_->MutRequestEntry(group.front())->desc().op_desc().backend();
  request_store_->ForEachMutRequestEntryForIdsInJob(
      group, [&](RequestEntry* request_entry, int32_t i, const RequestId& request_id) {
        CHECK_EQ(request_entry->desc().op_desc().backend(), backend);
      });
  return backend;
}

struct Scheduler::Impl {
  Impl();
  std::shared_ptr<RequestStore> request_store;
  std::shared_ptr<Executor> executor;
  std::shared_ptr<Coordinator> coordinator;
};

Scheduler::Impl::Impl() {
  request_store.reset(new RequestStore());
  executor.reset(new ExecutorImpl());
  executor->Init(request_store);
  coordinator.reset(new StaticGroupCoordinator());
  coordinator->Init(request_store, executor);
}

class SchedulerPlanToken {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SchedulerPlanToken);
  explicit SchedulerPlanToken(const std::vector<int64_t>& job_ids) : job_ids_(job_ids) {}
  ~SchedulerPlanToken() = default;
  const std::vector<int64_t>& job_ids() const { return job_ids_; }

 private:
  std::vector<int64_t> job_ids_;
};

SchedulerPlanToken* Scheduler::AddPlan(const Plan& plan) {
  std::vector<int64_t> job_ids;
  for (const auto& job_id7request_set : plan.collective_boxing_plan().job_id2request_set()) {
    const int64_t job_id = job_id7request_set.first;
    job_ids.emplace_back(job_id);
    impl_->request_store->InitJob(job_id, job_id7request_set.second);
    impl_->executor->InitJob(job_id);
    impl_->coordinator->InitJob(job_id);
  }
  return new SchedulerPlanToken(job_ids);
}

void Scheduler::DeletePlan(SchedulerPlanToken* plan_token) {
  const std::vector<int64_t>& job_ids = plan_token->job_ids();
  for (const auto& job_id : job_ids) {
    impl_->coordinator->DeinitJob(job_id);
    impl_->executor->DeinitJob(job_id);
    impl_->request_store->DeinitJob(job_id);
  }
  delete plan_token;
}

Scheduler::Scheduler() { impl_.reset(new Impl()); }

Scheduler::~Scheduler() = default;

RequestHandle* Scheduler::CreateRequestHandle(const RankDesc& rank_desc) {
  const RequestId& request_id =
      impl_->request_store->GetRequestIdByName(rank_desc.op_desc().name());
  auto* request_entry = impl_->request_store->MutRequestEntry(request_id);
  CHECK(rank_desc.op_desc() == request_entry->desc().op_desc());
  const int32_t local_rank = request_entry->GlobalRankToLocalRank(rank_desc.rank());
  void* request_entry_token = impl_->request_store->CreateRequestEntryToken(request_id);
  void* coordinator_token = impl_->coordinator->CreateCoordinatorToken(request_id);
  return new RequestHandle(local_rank, request_entry_token, coordinator_token);
}

void Scheduler::DestroyRequestHandle(RequestHandle* handle) {
  impl_->coordinator->DestroyCoordinatorToken(handle->coordinator_token());
  impl_->request_store->DestroyRequestEntryToken(handle->request_entry_token());
}

void Scheduler::Schedule(RequestHandle* handle,
                         std::shared_ptr<const RuntimeRequestInfo> request_info) {
  const int32_t local_rank = handle->local_rank();
  const bool ready = impl_->request_store->GetRequestEntry(handle->request_entry_token())
                         ->AddRuntimeRequest(local_rank, std::move(request_info));
  if (ready) { impl_->coordinator->AddRequest(handle->coordinator_token()); }
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
