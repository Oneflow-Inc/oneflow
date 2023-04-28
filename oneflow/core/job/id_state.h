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
#ifndef ONEFLOW_CORE_JOB_ID_STATE_H_
#define ONEFLOW_CORE_JOB_ID_STATE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/device/device_id.h"
#include "oneflow/core/graph/stream_id.h"
#include "oneflow/core/graph/task_id.h"

namespace oneflow {

class IdState {
 public:
  int64_t regst_desc_id_state_{};
  int64_t mem_block_id_state_{};
  int64_t chunk_id_state_{};
  int64_t job_id_state_{};
  HashMap<int64_t, uint32_t> task_index_state_{};
  HashMap<int64_t, uint32_t> stream_index_state_{};
};

class IdStateMgr final {
 public:
  IdStateMgr() = default;
  OF_DISALLOW_COPY_AND_MOVE(IdStateMgr);
  ~IdStateMgr() = default;

  IdState GetIdState();
  void SetIdState(const IdState& id_state);

  void SetRegstDescIdState(int64_t id) { id_state_.regst_desc_id_state_ = id; }
  int64_t GetRegstDescIdState() const { return id_state_.regst_desc_id_state_; }

  void SetMemBlockIdState(int64_t id) { id_state_.mem_block_id_state_ = id; }
  int64_t GetMemBlockIdState() const { return id_state_.mem_block_id_state_; }

  void SetChunkIdState(int64_t id) { id_state_.chunk_id_state_ = id; }
  int64_t GetChunkIdState() const { return id_state_.chunk_id_state_; }

  void SetJobIdState(int64_t id) { id_state_.job_id_state_ = id; }
  int64_t GetJobIdState() const { return id_state_.job_id_state_; }

  void SetTaskIndexState(const StreamId& stream_id, uint32_t idx) {
    id_state_.task_index_state_[EncodeStreamIdToInt64(stream_id)] = idx;
  }
  uint32_t GetTaskIndexState(const StreamId& stream_id) {
    auto encode_stream_id = EncodeStreamIdToInt64(stream_id);
    if (id_state_.task_index_state_.count(encode_stream_id) == 0) { return 0; }
    return id_state_.task_index_state_[encode_stream_id];
  }

  void SetStreamIndexState(const DeviceId device_id, uint32_t stream_index) {
    id_state_.stream_index_state_[EncodeDeviceIdToInt64(device_id)] = stream_index;
  }
  uint32_t GetStreamIndexState(const DeviceId& device_id) {
    auto encode_device_id = EncodeDeviceIdToInt64(device_id);
    if (id_state_.stream_index_state_.count(encode_device_id) == 0) { return 0; }
    return id_state_.stream_index_state_[encode_device_id];
  }

 private:
  IdState id_state_{};
};

}  // namespace oneflow

#endif
