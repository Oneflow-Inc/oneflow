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

namespace oneflow {

namespace boxing {

namespace collective {

namespace {

void SortRequestIdsByOrder(RequestStore* request_store, std::vector<int32_t>* requests) {
  std::sort(requests->begin(), requests->end(), [request_store](int32_t a, int32_t b) {
    return request_store->MutRequestEntry(a)->desc().order()
           < request_store->MutRequestEntry(b)->desc().order();
  });
}

}  // namespace

void StaticGroupCoordinator::Init(const CollectiveBoxingPlan& collective_boxing_plan,
                                  std::shared_ptr<RequestStore> request_store,
                                  std::shared_ptr<Executor> executor) {
  request_store_ = request_store;
  executor_ = executor;
  const CollectiveBoxingConf collective_boxing_conf =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  HashMap<int64_t, std::vector<int32_t>> job_id2request_ids;
  const int32_t request_count = request_store_->RequestCount();
  for (int32_t request_id = 0; request_id < request_count; ++request_id) {
    auto* request_entry = request_store_->MutRequestEntry(request_id);
    if (request_entry->HasRankOnThisNode()) {
      job_id2request_ids[request_entry->job_id()].push_back(request_id);
    }
  }
  const auto& GetRequestDesc = [&](int32_t request_id) -> const RequestDesc& {
    return request_store_->MutRequestEntry(request_id)->desc();
  };
  request_id2group_id_.resize(request_store_->RequestCount());
  request_id2index_in_group_.resize(request_store_->RequestCount());
  for (auto& job_id7request_ids : job_id2request_ids) {
    const int64_t job_id = job_id7request_ids.first;
    auto& request_ids = job_id7request_ids.second;
    SortRequestIdsByOrder(request_store_.get(), &request_ids);
    CHECK(std::adjacent_find(request_ids.begin(), request_ids.end(),
                             [&](int32_t a, int32_t b) {
                               return GetRequestDesc(a).dependency_depth()
                                      > GetRequestDesc(b).dependency_depth();
                             })
          == request_ids.end());
    executor_->GroupRequests(request_ids, [&](std::vector<int32_t>&& group) {
      const int32_t group_id = group_id2group_state_.size();
      group_id2group_state_.emplace_back(group.size());
      job_id2group_ids_[job_id].push_back(group_id);
      for (int32_t idx_in_group = 0; idx_in_group < group.size(); ++idx_in_group) {
        const int32_t request_id = group.at(idx_in_group);
        request_id2group_id_.at(request_id) = group_id;
        request_id2index_in_group_.at(request_id) = idx_in_group;
      }
      group_id2request_ids_.push_back(group);
    });
  }
  DumpSummary();
}

void StaticGroupCoordinator::AddRequest(int32_t request_id) {
  const int64_t job_id = request_store_->MutRequestEntry(request_id)->job_id();
  std::unique_lock<std::mutex> lock(mutex_);
  if (current_job_id_ == -1) {
    current_job_id_ = job_id;
    current_group_idx_in_job_ = 0;
  } else {
    CHECK_EQ(current_job_id_, job_id);
  }

  group_id2group_state_.at(request_id2group_id_.at(request_id))
      .AddReadyRequest(request_id2index_in_group_.at(request_id));
  const std::vector<int32_t>& group_ids = job_id2group_ids_.at(current_job_id_);
  int64_t num_launched_groups = 0;
  while (true) {
    const int32_t group_id = group_ids.at(current_group_idx_in_job_);
    auto& group_state = group_id2group_state_.at(group_id);
    if (group_state.IsReady()) {
      executor_->ExecuteGroupedRequests(group_id2request_ids_.at(group_id));
      group_state.Reset();
      current_group_idx_in_job_ = (current_group_idx_in_job_ + 1) % group_ids.size();
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

void StaticGroupCoordinator::DumpSummary() const {
  if (!Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) { return; }
  auto group_ls = TeePersistentLogStream::Create("boxing/collective/group");
  for (int32_t group_id = 0; group_id < group_id2group_state_.size(); ++group_id) {
    group_ls << "group id: " << std::to_string(group_id) << "\n";
    for (const int32_t request_id : group_id2request_ids_.at(group_id)) {
      group_ls->Write(request_store_->MutRequestEntry(request_id)->desc());
    }
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
