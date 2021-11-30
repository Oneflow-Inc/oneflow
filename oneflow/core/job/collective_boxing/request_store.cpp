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
#include "oneflow/core/job/collective_boxing/request_store.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace boxing {

namespace collective {

RequestEntry::RequestEntry(const RequestDesc& desc) : desc_(desc) {
  std::set<int64_t> node_ids;
  for (int64_t global_rank = 0; global_rank < desc.device_set().device().size(); ++global_rank) {
    const DeviceDesc& device = desc.device_set().device(global_rank);
    if (device.machine_id() == GlobalProcessCtx::Rank()) {
      local_device_vec_.emplace_back(device);
      global_rank2local_rank_.emplace(global_rank, local_rank2global_rank_.size());
      local_rank2global_rank_.emplace_back(global_rank);
    }
    node_ids.emplace(device.machine_id());
  }
  const size_t local_rank_count = local_device_vec_.size();
  node_count_ = node_ids.size();
  state_.runtime_request_info_vec.resize(local_rank_count);
  state_.runtime_request_count = 0;
  elem_cnt_ = Shape(desc.op_desc().shape()).elem_cnt();
  size_in_bytes_ = elem_cnt_ * GetSizeOfDataType(desc.op_desc().data_type());
  device_set_symbol_.reset(desc.device_set());
}

bool RequestEntry::AddRuntimeRequest(
    int32_t local_rank, std::shared_ptr<const RuntimeRequestInfo> runtime_request_info) {
  CHECK_LT(local_rank, state_.runtime_request_info_vec.size());
  std::lock_guard<std::mutex> lock(state_.mutex);
  CHECK(!state_.runtime_request_info_vec.at(local_rank));
  state_.runtime_request_info_vec.at(local_rank) = std::move(runtime_request_info);
  state_.runtime_request_count += 1;
  return state_.runtime_request_count == state_.runtime_request_info_vec.size();
}

const std::shared_ptr<const RuntimeRequestInfo>& RequestEntry::GetRuntimeRequest(
    int32_t local_rank) {
  std::lock_guard<std::mutex> lock(state_.mutex);
  return state_.runtime_request_info_vec.at(local_rank);
}

std::vector<std::shared_ptr<const RuntimeRequestInfo>> RequestEntry::ResetRuntimeRequest() {
  std::lock_guard<std::mutex> lock(state_.mutex);
  std::vector<std::shared_ptr<const RuntimeRequestInfo>> ret(
      state_.runtime_request_info_vec.size());
  ret.swap(state_.runtime_request_info_vec);
  state_.runtime_request_count = 0;
  return ret;
}

void RequestStore::InitJob(int64_t job_id, const RequestSet& request_set) {
  std::vector<std::unique_ptr<RequestEntry>>& request_entry_vec = job_id2request_entry_vec_[job_id];
  CHECK_EQ(request_entry_vec.size(), 0);
  for (const RequestDesc& desc : request_set.request()) {
    request_entry_vec.emplace_back(std::make_unique<RequestEntry>(desc));
  }
  for (int32_t i = 0; i < request_entry_vec.size(); ++i) {
    const std::unique_ptr<RequestEntry>& entry = request_entry_vec.at(i);
    CHECK(name2request_id_.emplace(entry->desc().op_desc().name(), RequestId(job_id, i)).second);
  }
}

void RequestStore::DeinitJob(int64_t job_id) {
  const auto& it = job_id2request_entry_vec_.find(job_id);
  CHECK(it != job_id2request_entry_vec_.end());
  const auto& request_entry_vec = it->second;
  for (const auto& request_entry : request_entry_vec) {
    name2request_id_.erase(request_entry->desc().op_desc().name());
  }
  job_id2request_entry_vec_.erase(job_id);
}

struct RequestEntryToken {
  RequestEntry* request_entry;
};

void* RequestStore::CreateRequestEntryToken(const RequestId& request_id) {
  auto it = job_id2request_entry_vec_.find(request_id.job_id);
  CHECK(it != job_id2request_entry_vec_.end());
  return new RequestEntryToken{it->second.at(request_id.request_index).get()};
}

void RequestStore::DestroyRequestEntryToken(void* request_entry_token) {
  auto token = static_cast<RequestEntryToken*>(request_entry_token);
  delete token;
}

RequestEntry* RequestStore::GetRequestEntry(void* request_entry_token) {
  return static_cast<RequestEntryToken*>(request_entry_token)->request_entry;
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
