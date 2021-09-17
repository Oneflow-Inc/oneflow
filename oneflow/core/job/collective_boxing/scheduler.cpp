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
  RequestHandle(int32_t local_rank, void* request_entry_token, void* request_token,
                void* executor_token)
      : local_rank_(local_rank),
        request_entry_token_(request_entry_token),
        request_token_(request_token),
        executor_token_(executor_token) {}
  ~RequestHandle() = default;

  int32_t local_rank() const { return local_rank_; }

  void* request_entry_token() { return request_entry_token_; }

  void* request_token() { return request_token_; }

  void* executor_token() { return executor_token_; }

 private:
  int32_t local_rank_;
  void* request_entry_token_;
  void* request_token_;
  void* executor_token_;
};

class ExecutorImpl : public Executor {
 public:
  ExecutorImpl() = default;
  ~ExecutorImpl() override = default;

  void Init(std::shared_ptr<RequestStore> request_store) override;
  void InitJob(int64_t job_id) override;
  void DeinitJob(int64_t job_id) override;
  void GroupRequests(const std::vector<RequestId>& request_ids,
                     const std::function<void(std::vector<RequestId>&&)>& Handler) override;
  void ExecuteGroupedRequests(const std::vector<RequestId>& request_ids,
                              void* executor_token) override;
  void* CreateExecutorToken(const RequestId& request_id) override;
  void DestroyExecutorToken(void* executor_token) override;

 private:
  Backend GetUniqueBackend(const std::vector<RequestId>& request_ids);

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

void* ExecutorImpl::CreateExecutorToken(const RequestId& request_id) {
  return backends_.at(Backend::kBackendNCCL)->CreateExecutorToken(request_id);
}

void ExecutorImpl::DestroyExecutorToken(void* executor_token) {
  backends_.at(Backend::kBackendNCCL)->DestroyExecutorToken(executor_token);
}

void ExecutorImpl::GroupRequests(const std::vector<RequestId>& request_ids,
                                 const std::function<void(std::vector<RequestId>&&)>& Handler) {
  if (request_ids.empty()) { return; }
  const CollectiveBoxingConf& conf =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  if (!conf.enable_fusion()) {
    for (auto request_id : request_ids) { Handler(std::vector<RequestId>({request_id})); }
    return;
  }
  auto HandleGroup = [&]() {
    if (group_buffer_.empty()) { return; }
    if (group_buffer_.size() == 1) {
      Handler(std::vector<RequestId>({group_buffer_.front()}));
    } else {
      const auto backend =
          request_store_->MutRequestEntry(group_buffer_.front())->desc().op_desc().backend();
      backends_.at(backend)->GroupRequests(group_buffer_, Handler);
    }
    group_buffer_.clear();
  };
  request_store_->ForEachMutRequestEntryForIdsInJob(
      request_ids, [&](RequestEntry* request_entry, int32_t i, RequestId request_id) {
        const int64_t job_id = request_id.job_id;
        if (!group_buffer_.empty()) {
          const auto* cur_entry = request_entry;
          const auto* group_entry = request_store_->MutRequestEntry(group_buffer_.front());
          if (job_id != group_buffer_job_id_
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

void ExecutorImpl::ExecuteGroupedRequests(const std::vector<RequestId>& request_ids,
                                          void* executor_token) {
  if (request_ids.empty()) { return; }
  const Backend backend = GetUniqueBackend(request_ids);
  backends_.at(backend)->ExecuteRequests(request_ids, executor_token);
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
  const CollectiveBoxingConf& conf =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  request_store.reset(new RequestStore());
  executor.reset(new ExecutorImpl());
  executor->Init(request_store);
  if (conf.has_static_group_coordinator_conf()
      || conf.coordinator_conf_case() == CollectiveBoxingConf::COORDINATOR_CONF_NOT_SET) {
    coordinator.reset(new StaticGroupCoordinator());
  } else {
    UNIMPLEMENTED();
  }
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
  void* request_token = impl_->coordinator->CreateRequestToken(request_id);
  void* executor_token = impl_->executor->CreateExecutorToken(request_id);
  return new RequestHandle(local_rank, request_entry_token, request_token, executor_token);
}

void Scheduler::DestroyRequestHandle(RequestHandle* handle) {
  impl_->executor->DestroyExecutorToken(handle->executor_token());
  impl_->coordinator->DestroyRequestToken(handle->request_token());
  impl_->request_store->DestroyRequestEntryToken(handle->request_entry_token());
}

void Scheduler::Schedule(RequestHandle* handle,
                         std::shared_ptr<const RuntimeRequestInfo> request_info) {
  const int32_t local_rank = handle->local_rank();
  const bool ready = impl_->request_store->GetRequestEntry(handle->request_entry_token())
                         ->AddRuntimeRequest(local_rank, std::move(request_info));
  if (ready) { impl_->coordinator->AddRequest(handle->request_token(), handle->executor_token()); }
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
