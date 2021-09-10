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

void SortRequestIdsByOrder(const int64_t job_id, RequestStore* request_store,
                           std::vector<int32_t>* requests) {
  std::sort(requests->begin(), requests->end(), [job_id, request_store](int32_t a, int32_t b) {
    return request_store->MutRequestEntry(job_id, a)->desc().order()
           < request_store->MutRequestEntry(job_id, b)->desc().order();
  });
}

}  // namespace

void StaticGroupCoordinator::Init(std::shared_ptr<RequestStore> request_store,
                                  std::shared_ptr<Executor> executor) {
  request_store_ = request_store;
  executor_ = executor;
}

void* StaticGroupCoordinator::CreateRequestToken(int64_t job_id, int32_t request_id) {
  auto it = job_id2static_group_requests_info_.find(job_id);
  CHECK(it != job_id2static_group_requests_info_.end());
  return new StaticGroupRequestsInfoToken{job_id, request_id, &it->second};
}

void StaticGroupCoordinator::DestroyRequestToken(void* request_token) {
  auto token = static_cast<StaticGroupRequestsInfoToken*>(request_token);
  delete token;
}

void StaticGroupCoordinator::AddPlan(const std::vector<int64_t>& job_ids) {
  const auto& GetRequestDesc = [&](int64_t job_id, int32_t request_id) -> const RequestDesc& {
    return request_store_->MutRequestEntry(job_id, request_id)->desc();
  };

  for (const auto& job_id : job_ids) {
    std::vector<int32_t> request_ids;
    request_store_->ForEachMutRequestEntryInJob(
        job_id, [&](RequestEntry* request_entry, int32_t i, int32_t request_id) {
          if (request_entry->HasRankOnThisNode()) { request_ids.push_back(request_id); }
        });
    SortRequestIdsByOrder(job_id, request_store_.get(), &request_ids);
    CHECK(std::adjacent_find(request_ids.begin(), request_ids.end(),
                             [&](int32_t a, int32_t b) {
                               return GetRequestDesc(job_id, a).dependency_depth()
                                      > GetRequestDesc(job_id, b).dependency_depth();
                             })
          == request_ids.end());
    StaticGroupRequestsInfo info;
    std::vector<GroupState>& group_states = info.group_states;
    std::vector<RequestIndex>& request_id2index = info.request_id2index;
    std::vector<std::vector<int32_t>>& group_id2request_ids = info.group_id2request_ids;
    const int32_t request_count = request_store_->RequestCount4Job(job_id);
    request_id2index.resize(request_count);
    executor_->GroupRequests(
        job_id, request_ids, [&](int64_t job_id, std::vector<int32_t>&& group) {
          const int32_t group_id = group_states.size();
          group_states.emplace_back(group.size());
          for (int32_t idx_in_group = 0; idx_in_group < group.size(); ++idx_in_group) {
            const int32_t request_id = group.at(idx_in_group);
            request_id2index.at(request_id).group_id = group_id;
            request_id2index.at(request_id).index_in_group = idx_in_group;
          }
          group_id2request_ids.push_back(group);
        });
    CHECK(job_id2static_group_requests_info_.emplace(job_id, info).second);
    if (group_states.size() != 0) { DumpSummary(job_id); }
  }
}

void StaticGroupCoordinator::DeletePlan(const std::vector<int64_t>& job_ids) {
  for (const auto& job_id : job_ids) { job_id2static_group_requests_info_.erase(job_id); }
}

void StaticGroupCoordinator::AddRequest(void* request_token, void* executor_token) {
  std::unique_lock<std::mutex> lock(mutex_);
  StaticGroupRequestsInfoToken* token = static_cast<StaticGroupRequestsInfoToken*>(request_token);
  if (current_job_id_ == -1) {
    current_job_id_ = token->job_id;
    current_group_idx_in_job_ = 0;
  } else {
    CHECK_EQ(current_job_id_, token->job_id);
  }
  const int32_t request_id = token->request_id;
  StaticGroupRequestsInfo* info = token->info;
  info->group_states.at(info->request_id2index.at(request_id).group_id)
      .AddReadyRequest(info->request_id2index.at(request_id).index_in_group);
  int64_t num_launched_groups = 0;
  while (true) {
    auto& group_state = info->group_states.at(current_group_idx_in_job_);
    if (group_state.IsReady()) {
      executor_->ExecuteGroupedRequests(current_job_id_,
                                        info->group_id2request_ids.at(current_group_idx_in_job_),
                                        executor_token);
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
  if (!Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) { return; }
  auto group_ls = TeePersistentLogStream::Create(StrCat("boxing/collective/job_", job_id));
  const auto& it = job_id2static_group_requests_info_.find(job_id);

  CHECK(it != job_id2static_group_requests_info_.end());
  const auto& group_id2request_ids = it->second.group_id2request_ids;
  for (int32_t group_id = 0; group_id < group_id2request_ids.size(); ++group_id) {
    group_ls << "group id: " << std::to_string(group_id) << "\n";
    request_store_->ForEachMutRequestEntryForIdsInJob(
        job_id, group_id2request_ids.at(group_id),
        [&](RequestEntry* request_entry, int32_t i, int32_t request_id) {
          group_ls->Write(request_entry->desc());
        });
  }
}

void StaticGroupCoordinator::GroupState::AddReadyRequest(int32_t index) {
  CHECK(!index2is_ready.at(index));
  CHECK(index2is_ready.at(index) = true);
  ready_request_count += 1;
}

bool StaticGroupCoordinator::GroupState::IsReady() const {
  return ready_request_count == index2is_ready.size();
}

void StaticGroupCoordinator::GroupState::Reset() {
  ready_request_count = 0;
  std::fill(index2is_ready.begin(), index2is_ready.end(), false);
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
