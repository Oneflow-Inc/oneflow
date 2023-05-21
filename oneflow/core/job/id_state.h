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

}  // namespace oneflow

#endif
