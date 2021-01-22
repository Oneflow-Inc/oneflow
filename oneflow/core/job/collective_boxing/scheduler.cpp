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
#ifdef WITH_MPI
#include "oneflow/core/job/collective_boxing/dynamic_coordinator.h"
#endif

namespace oneflow {

namespace boxing {

namespace collective {

class RequestHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RequestHandle)
  RequestHandle(int32_t request_id, int32_t local_rank)
      : request_id_(request_id), local_rank_(local_rank) {}
  ~RequestHandle() = default;

  int32_t request_id() const { return request_id_; }

  int32_t local_rank() const { return local_rank_; }

 private:
  int32_t request_id_;
  int32_t local_rank_;
};

class ExecutorImpl : public Executor {
 public:
  ExecutorImpl() = default;
  ~ExecutorImpl() override = default;

  void Init(const CollectiveBoxingPlan& collective_boxing_plan,
            std::shared_ptr<RequestStore> request_store) override;
  void GroupRequests(const std::vector<int32_t>& request_ids,
                     const std::function<void(std::vector<int32_t>&&)>& Handler) override;
  void ExecuteGroupedRequests(const std::vector<int32_t>& request_ids) override;

 private:
  Backend GetUniqueBackend(const std::vector<int32_t>& request_ids);

  std::vector<std::unique_ptr<ExecutorBackend>> backends_;
  std::shared_ptr<RequestStore> request_store_;
  std::vector<int32_t> group_buffer_;
};

void ExecutorImpl::Init(const CollectiveBoxingPlan& collective_boxing_plan,
                        std::shared_ptr<RequestStore> request_store) {
  request_store_ = request_store;
  backends_.resize(Backend_ARRAYSIZE);
#ifdef WITH_CUDA
  std::unique_ptr<ExecutorBackend> nccl_backend = std::make_unique<NcclExecutorBackend>();
  nccl_backend->Init(collective_boxing_plan, request_store_);
  backends_.at(Backend::kBackendNCCL) = std::move(nccl_backend);
#endif
}

void ExecutorImpl::GroupRequests(const std::vector<int32_t>& request_ids,
                                 const std::function<void(std::vector<int32_t>&&)>& Handler) {
  if (request_ids.empty()) { return; }
  const CollectiveBoxingConf& conf =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  if (!conf.enable_fusion()) {
    for (int32_t request_id : request_ids) { Handler(std::vector<int32_t>({request_id})); }
    return;
  }
  auto HandleGroup = [&]() {
    if (group_buffer_.empty()) { return; }
    if (group_buffer_.size() == 1) {
      Handler(std::vector<int32_t>({group_buffer_.front()}));
    } else {
      const auto backend =
          request_store_->MutRequestEntry(group_buffer_.front())->desc().op_desc().backend();
      backends_.at(backend)->GroupRequests(group_buffer_, Handler);
    }
    group_buffer_.clear();
  };
  for (const int32_t request_id : request_ids) {
    if (!group_buffer_.empty()) {
      const auto* cur_entry = request_store_->MutRequestEntry(request_id);
      const auto* group_entry = request_store_->MutRequestEntry(group_buffer_.front());
      if (cur_entry->desc().dependency_depth() != group_entry->desc().dependency_depth()
          || cur_entry->desc().op_desc().backend() != group_entry->desc().op_desc().backend()
          || cur_entry->device_set_symbol() != group_entry->device_set_symbol()) {
        HandleGroup();
      }
    }
    group_buffer_.push_back(request_id);
  }
  HandleGroup();
}

void ExecutorImpl::ExecuteGroupedRequests(const std::vector<int32_t>& request_ids) {
  if (request_ids.empty()) { return; }
  const Backend backend = GetUniqueBackend(request_ids);
  backends_.at(backend)->ExecuteRequests(request_ids);
}

Backend ExecutorImpl::GetUniqueBackend(const std::vector<int32_t>& request_ids) {
  const Backend backend =
      request_store_->MutRequestEntry(request_ids.front())->desc().op_desc().backend();
  for (int32_t i = 1; i < request_ids.size(); ++i) {
    CHECK_EQ(request_store_->MutRequestEntry(request_ids.at(i))->desc().op_desc().backend(),
             backend);
  }
  return backend;
}

struct Scheduler::Impl {
  explicit Impl(const CollectiveBoxingPlan& collective_boxing_plan);

  CollectiveBoxingPlan collective_boxing_plan;
  std::shared_ptr<RequestStore> request_store;
  std::shared_ptr<Coordinator> coordinator;
};

Scheduler::Impl::Impl(const CollectiveBoxingPlan& collective_boxing_plan)
    : collective_boxing_plan(collective_boxing_plan) {
  const CollectiveBoxingConf& conf =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  request_store.reset(new RequestStore(collective_boxing_plan));
  std::shared_ptr<Executor> executor(new ExecutorImpl());
  executor->Init(collective_boxing_plan, request_store);
  if (conf.has_static_group_coordinator_conf()
      || conf.coordinator_conf_case() == CollectiveBoxingConf::COORDINATOR_CONF_NOT_SET) {
    coordinator.reset(new StaticGroupCoordinator());
  } else if (conf.has_dynamic_coordinator_conf()) {
#ifdef WITH_MPI
    coordinator.reset(new DynamicCoordinator());
#else
    LOG(FATAL) << "MPI components not found";
#endif
  } else {
    UNIMPLEMENTED();
  }
  coordinator->Init(collective_boxing_plan, request_store, executor);
}

Scheduler::Scheduler(const Plan& plan) { impl_.reset(new Impl(plan.collective_boxing_plan())); }

Scheduler::~Scheduler() = default;

std::shared_ptr<RequestHandle> Scheduler::CreateRequestHandle(const RankDesc& rank_desc) {
  const int32_t request_id = impl_->request_store->GetRequestIdByName(rank_desc.op_desc().name());
  auto* request_entry = impl_->request_store->MutRequestEntry(request_id);
  CHECK(rank_desc.op_desc() == request_entry->desc().op_desc());
  const int32_t local_rank = request_entry->GlobalRankToLocalRank(rank_desc.rank());
  return std::make_shared<RequestHandle>(request_id, local_rank);
}

void Scheduler::Schedule(const std::shared_ptr<RequestHandle>& handle,
                         std::shared_ptr<const RuntimeRequestInfo> request_info) {
  const int32_t request_id = handle->request_id();
  const int32_t local_rank = handle->local_rank();
  const bool ready = impl_->request_store->MutRequestEntry(request_id)
                         ->AddRuntimeRequest(local_rank, std::move(request_info));
  if (ready) { impl_->coordinator->AddRequest(request_id); }
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
