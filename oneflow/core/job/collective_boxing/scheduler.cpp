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

namespace oneflow {

namespace boxing {

namespace collective {

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

class ExecutorToken final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecutorToken);
  ExecutorToken(Backend backend, void* backend_token)
      : backend_(backend), backend_token_(backend_token) {}
  ~ExecutorToken() = default;

  Backend backend() { return backend_; }

  void* backend_token() { return backend_token_; }

 private:
  Backend backend_;
  void* backend_token_;
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
      const std::function<void(std::vector<RequestId>&&, ExecutorToken*)>& Handler) override;
  void ExecuteGroupedRequests(ExecutorToken* executor_token) override;
  void DestroyExecutorToken(ExecutorToken* executor_token) override;

 private:
  Backend GetUniqueBackend(const std::vector<RequestId>& request_ids);
  ExecutorToken* CreateExecutorToken(const std::vector<RequestId>& request_ids,
                                     void* backend_token);

  std::vector<std::unique_ptr<ExecutorBackend>> backends_;
  std::shared_ptr<RequestStore> request_store_;
  std::vector<RequestId> group_buffer_;
  int64_t group_buffer_job_id_{};
};

void ExecutorImpl::Init(std::shared_ptr<RequestStore> request_store) {
  request_store_ = request_store;
  backends_.resize(Backend_ARRAYSIZE);
#ifdef WITH_CUDA
  std::unique_ptr<ExecutorBackend> nccl_backend = std::make_unique<NcclExecutorBackend>();
  nccl_backend->Init(request_store_);
  backends_.at(Backend::kBackendNCCL) = std::move(nccl_backend);
#endif
}

void ExecutorImpl::InitJob(int64_t job_id) { backends_.at(Backend::kBackendNCCL)->InitJob(job_id); }

void ExecutorImpl::DeinitJob(int64_t job_id) {
  backends_.at(Backend::kBackendNCCL)->DeinitJob(job_id);
}

ExecutorToken* ExecutorImpl::CreateExecutorToken(const std::vector<RequestId>& request_ids,
                                                 void* backend_token) {
  return new ExecutorToken(GetUniqueBackend(request_ids), backend_token);
}

void ExecutorImpl::DestroyExecutorToken(ExecutorToken* executor_token) {
  backends_.at(Backend::kBackendNCCL)->DestroyExecutorBackendToken(executor_token->backend_token());
  delete executor_token;
}

void ExecutorImpl::GroupRequests(
    const std::vector<RequestId>& request_ids,
    const std::function<void(std::vector<RequestId>&&, ExecutorToken*)>& Handler) {
  if (request_ids.empty()) { return; }
  const CollectiveBoxingConf& conf =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  auto BackendHandler = [&](std::vector<RequestId>&& group, void* backend_token) {
    ExecutorToken* executor_token = CreateExecutorToken(group, backend_token);
    Handler(std::move(group), executor_token);
  };
  auto HandleGroup = [&]() {
    if (group_buffer_.empty()) { return; }
    const auto backend =
        request_store_->MutRequestEntry(group_buffer_.front())->desc().op_desc().backend();
    backends_.at(backend)->GroupRequests(group_buffer_, BackendHandler);
    group_buffer_.clear();
  };
  request_store_->ForEachMutRequestEntryForIdsInJob(
      request_ids, [&](RequestEntry* request_entry, int32_t i, RequestId request_id) {
        const int64_t job_id = request_id.job_id;
        if (!group_buffer_.empty()) {
          const auto* cur_entry = request_entry;
          const auto* group_entry = request_store_->MutRequestEntry(group_buffer_.front());
          if ((!conf.enable_fusion()) || job_id != group_buffer_job_id_
              || cur_entry->desc().dependency_depth() != group_entry->desc().dependency_depth()
              || cur_entry->desc().op_desc().backend() != group_entry->desc().op_desc().backend()
              || cur_entry->device_set_symbol() != group_entry->device_set_symbol()) {
            HandleGroup();
          }
        }
        group_buffer_.push_back(request_id);
        group_buffer_job_id_ = job_id;
      });
  HandleGroup();
}

void ExecutorImpl::ExecuteGroupedRequests(ExecutorToken* executor_token) {
  const Backend backend = executor_token->backend();
  backends_.at(backend)->ExecuteRequests(executor_token->backend_token());
}

Backend ExecutorImpl::GetUniqueBackend(const std::vector<RequestId>& request_ids) {
  const Backend backend =
      request_store_->MutRequestEntry(request_ids.front())->desc().op_desc().backend();
  request_store_->ForEachMutRequestEntryForIdsInJob(
      request_ids, [&](RequestEntry* request_entry, int32_t i, RequestId request_id) {
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
    job_ids.push_back(job_id);
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
