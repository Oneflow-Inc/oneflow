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
#include "oneflow/core/job/collective_boxing/static_group_coordinator.h"
#include "oneflow/core/job/collective_boxing/executor.h"
#include "oneflow/core/job/collective_boxing/request_store.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

namespace boxing {

namespace collective {

namespace {

void SortRequestIdsByOrder(RequestStore* request_store, std::vector<RequestId>* requests) {
  std::sort(requests->begin(), requests->end(),
            [request_store](const RequestId& a, const RequestId& b) {
              return request_store->MutRequestEntry(a)->desc().order()
                     < request_store->MutRequestEntry(b)->desc().order();
            });
}

bool HasRankInteractionOnDeviceSet(const DeviceSet& a, const DeviceSet& b) {
  for (int64_t i = 0; i < a.device_size(); ++i) {
    const DeviceDesc& a_device_desc = a.device(i);
    for (int64_t j = 0; j < b.device_size(); ++j) {
      if (a_device_desc.machine_id() == b.device(j).machine_id()) { return true; }
    }
  }
  return false;
}

}  // namespace

struct GroupState {
  explicit GroupState(int32_t group_size) : index2is_ready(group_size), ready_request_count(0) {}

  void AddReadyRequest(int32_t index);
  bool IsReady() const;
  void Reset();

  std::vector<bool> index2is_ready;
  int32_t ready_request_count;
};
std::mutex mutex_;
int64_t current_job_id_ = -1;
int64_t current_group_idx_in_job_ = -1;

struct RequestGroupIndex {
  int32_t group_id;
  int32_t index_in_group;
};

class GroupToken;

struct StaticGroupRequestsInfo {
  std::vector<RequestGroupIndex> request_index2request_group_index;
  std::vector<GroupState> group_states;
  std::vector<std::vector<RequestId>> group_id2request_ids;
  std::vector<GroupToken*> group_id2group_token;
};

struct StaticGroupRequestsInfoToken {
  RequestId request_id;
  StaticGroupRequestsInfo* info;
};

struct StaticGroupCoordinator::Impl {
  Impl(const std::shared_ptr<RequestStore>& request_store,
       const std::shared_ptr<Executor>& executor);
  std::shared_ptr<RequestStore> request_store_;
  std::shared_ptr<Executor> executor_;
  HashMap<int64_t, StaticGroupRequestsInfo> job_id2static_group_requests_info_;
};

StaticGroupCoordinator::Impl::Impl(const std::shared_ptr<RequestStore>& request_store,
                                   const std::shared_ptr<Executor>& executor)
    : request_store_(request_store), executor_(executor) {}

StaticGroupCoordinator::StaticGroupCoordinator() = default;

StaticGroupCoordinator::~StaticGroupCoordinator() = default;

void StaticGroupCoordinator::Init(std::shared_ptr<RequestStore> request_store,
                                  std::shared_ptr<Executor> executor) {
  impl_ = std::make_unique<Impl>(request_store, executor);
}

void* StaticGroupCoordinator::CreateCoordinatorToken(const RequestId& request_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = impl_->job_id2static_group_requests_info_.find(request_id.job_id);
  CHECK(it != impl_->job_id2static_group_requests_info_.end());
  return new StaticGroupRequestsInfoToken{request_id, &it->second};
}

void StaticGroupCoordinator::DestroyCoordinatorToken(void* coordinator_token) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto token = static_cast<StaticGroupRequestsInfoToken*>(coordinator_token);
  delete token;
}

void StaticGroupCoordinator::InitJob(int64_t job_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  std::vector<RequestId> request_ids;
  impl_->request_store_->ForEachMutRequestEntryInJob(
      job_id, [&](RequestEntry* request_entry, int32_t i, const RequestId& request_id) {
        request_ids.emplace_back(request_id);
      });
  SortRequestIdsByOrder(impl_->request_store_.get(), &request_ids);
  StaticGroupRequestsInfo info;
  std::vector<GroupState>& group_states = info.group_states;
  std::vector<RequestGroupIndex>& request_index2request_group_index =
      info.request_index2request_group_index;
  std::vector<std::vector<RequestId>>& group_id2request_ids = info.group_id2request_ids;
  std::vector<GroupToken*>& group_id2group_token = info.group_id2group_token;
  const int32_t request_count = impl_->request_store_->RequestCountForJob(job_id);
  request_index2request_group_index.resize(request_count);
  impl_->executor_->GroupRequests(
      request_ids, [&](std::vector<RequestId>&& group, GroupToken* group_token) {
        const int32_t group_id = group_states.size();
        group_states.emplace_back(group.size());
        for (int32_t idx_in_group = 0; idx_in_group < group.size(); ++idx_in_group) {
          const RequestId& request_id = group.at(idx_in_group);
          RequestGroupIndex request_group_index{group_id, idx_in_group};
          request_index2request_group_index.at(request_id.request_index) = request_group_index;
        }
        group_id2request_ids.emplace_back(group);
        group_id2group_token.emplace_back(group_token);
      });

  CHECK(impl_->job_id2static_group_requests_info_.emplace(job_id, info).second);
  if (group_states.size() != 0) { DumpSummary(job_id); }
}

void StaticGroupCoordinator::DeinitJob(int64_t job_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  const auto& it = impl_->job_id2static_group_requests_info_.find(job_id);
  CHECK(it != impl_->job_id2static_group_requests_info_.end());
  const auto& group_id2group_token = it->second.group_id2group_token;
  for (int32_t group_id = 0; group_id < group_id2group_token.size(); ++group_id) {
    impl_->executor_->DestroyGroupToken(group_id2group_token.at(group_id));
  }
  impl_->job_id2static_group_requests_info_.erase(job_id);
}

void StaticGroupCoordinator::AddRequest(void* coordinator_token) {
  std::unique_lock<std::mutex> lock(mutex_);
  StaticGroupRequestsInfoToken* token =
      static_cast<StaticGroupRequestsInfoToken*>(coordinator_token);
  const RequestId& request_id = token->request_id;
  if (current_job_id_ == -1) {
    current_job_id_ = request_id.job_id;
    current_group_idx_in_job_ = 0;
  } else {
    CHECK_EQ(current_job_id_, request_id.job_id);
  }
  StaticGroupRequestsInfo* info = token->info;
  const RequestGroupIndex& request_group_index =
      info->request_index2request_group_index.at(request_id.request_index);
  info->group_states.at(request_group_index.group_id)
      .AddReadyRequest(request_group_index.index_in_group);
  int64_t num_launched_groups = 0;
  while (true) {
    auto& group_state = info->group_states.at(current_group_idx_in_job_);
    if (group_state.IsReady()) {
      impl_->executor_->ExecuteGroup(info->group_id2group_token.at(current_group_idx_in_job_));
      group_state.Reset();
      current_group_idx_in_job_ = (current_group_idx_in_job_ + 1) % info->group_states.size();
      num_launched_groups += 1;
    } else {
      break;
    }
  }
  if (current_group_idx_in_job_ == 0 && num_launched_groups > 0) {
    current_job_id_ = -1;
    current_group_idx_in_job_ = -1;
  }
}

void StaticGroupCoordinator::DumpSummary(const int64_t job_id) const {
  if (!Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) { return; }
  auto group_ls = TeePersistentLogStream::Create(StrCat("boxing/collective/job_", job_id));
  const auto& it = impl_->job_id2static_group_requests_info_.find(job_id);

  CHECK(it != impl_->job_id2static_group_requests_info_.end());
  const auto& group_id2request_ids = it->second.group_id2request_ids;
  for (int32_t group_id = 0; group_id < group_id2request_ids.size(); ++group_id) {
    group_ls << "group id: " << std::to_string(group_id) << "\n";
    impl_->request_store_->ForEachMutRequestEntryForIdsInJob(
        group_id2request_ids.at(group_id),
        [&](RequestEntry* request_entry, int32_t i, const RequestId& request_id) {
          group_ls->Write(request_entry->desc());
        });
  }
}

void GroupState::AddReadyRequest(int32_t index) {
  CHECK(!index2is_ready.at(index));
  CHECK(index2is_ready.at(index) = true);
  ready_request_count += 1;
}

bool GroupState::IsReady() const { return ready_request_count == index2is_ready.size(); }

void GroupState::Reset() {
  ready_request_count = 0;
  std::fill(index2is_ready.begin(), index2is_ready.end(), false);
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
