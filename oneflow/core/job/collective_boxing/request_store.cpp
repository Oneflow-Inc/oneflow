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
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace boxing {

namespace collective {

RequestEntry::RequestEntry(int64_t job_id, const RequestDesc& desc) : job_id_(job_id), desc_(desc) {
  std::set<int64_t> node_ids;
  for (int64_t global_rank = 0; global_rank < desc.device_set().device().size(); ++global_rank) {
    const DeviceDesc& device = desc.device_set().device(global_rank);
    if (device.machine_id() == Global<MachineCtx>::Get()->this_machine_id()) {
      local_device_vec_.push_back(device);
      global_rank2local_rank_.emplace(global_rank, local_rank2global_rank_.size());
      local_rank2global_rank_.push_back(global_rank);
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

RequestStore::RequestStore(const CollectiveBoxingPlan& collective_boxing_plan) {
  for (const auto& job_id7request_set : collective_boxing_plan.job_id2request_set()) {
    const int64_t job_id = job_id7request_set.first;
    const RequestSet& request_set = job_id7request_set.second;
    for (const RequestDesc& desc : request_set.request()) {
      request_entry_vec_.emplace_back(std::make_unique<RequestEntry>(job_id, desc));
    }
  }
  std::sort(request_entry_vec_.begin(), request_entry_vec_.end(),
            [](const std::unique_ptr<RequestEntry>& a, const std::unique_ptr<RequestEntry>& b) {
              return a->NodeCount() == a->NodeCount()
                         ? a->desc().op_desc().name() < b->desc().op_desc().name()
                         : a->NodeCount() > a->NodeCount();
            });
  max_multi_node_request_id_ = 0;
  for (int32_t i = 0; i < request_entry_vec_.size(); ++i) {
    const std::unique_ptr<RequestEntry>& entry = request_entry_vec_.at(i);
    CHECK(name2request_id_.emplace(entry->desc().op_desc().name(), i).second);
    if (entry->NodeCount() > 1) { max_multi_node_request_id_ = i + 1; }
  }
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
