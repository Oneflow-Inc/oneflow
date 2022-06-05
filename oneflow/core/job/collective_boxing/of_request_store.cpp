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
#include "oneflow/core/job/collective_boxing/of_request_store.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace boxing {

namespace collective {

OfRequestEntry::OfRequestEntry(const RequestDesc& desc) : desc_(desc) {
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
  elem_cnt_ = Shape(desc.op_desc().shape()).elem_cnt();
  size_in_bytes_ = elem_cnt_ * GetSizeOfDataType(desc.op_desc().data_type());
  device_set_symbol_.reset(desc.device_set());
}

void OfRequestStore::InitJob(int64_t job_id, const RequestSet& request_set) {
  std::vector<std::unique_ptr<OfRequestEntry>>& request_entry_vec = job_id2request_entry_vec_[job_id];
  CHECK_EQ(request_entry_vec.size(), 0);
  for (const RequestDesc& desc : request_set.request()) {
    request_entry_vec.emplace_back(std::make_unique<OfRequestEntry>(desc));
  }
  for (int32_t i = 0; i < request_entry_vec.size(); ++i) {
    const std::unique_ptr<OfRequestEntry>& entry = request_entry_vec.at(i);
    CHECK(name2request_id_.emplace(entry->desc().op_desc().name(), OfRequestId(job_id, i)).second);
  }
}

void OfRequestStore::DeinitJob(int64_t job_id) {
  const auto& it = job_id2request_entry_vec_.find(job_id);
  CHECK(it != job_id2request_entry_vec_.end());
  const auto& request_entry_vec = it->second;
  for (const auto& request_entry : request_entry_vec) {
    name2request_id_.erase(request_entry->desc().op_desc().name());
  }
  job_id2request_entry_vec_.erase(job_id);
}

struct OfRequestEntryToken {
  OfRequestEntry* request_entry;
};

void* OfRequestStore::CreateOfRequestEntryToken(const OfRequestId& request_id) {
  auto it = job_id2request_entry_vec_.find(request_id.job_id);
  CHECK(it != job_id2request_entry_vec_.end());
  return new OfRequestEntryToken{it->second.at(request_id.request_index).get()};
}

void OfRequestStore::DestroyOfRequestEntryToken(void* request_entry_token) {
  auto token = static_cast<OfRequestEntryToken*>(request_entry_token);
  delete token;
}

OfRequestEntry* OfRequestStore::GetOfRequestEntry(void* request_entry_token) {
  return static_cast<OfRequestEntryToken*>(request_entry_token)->request_entry;
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
