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
#include "oneflow/core/job/id_manager.h"
#include <string>

namespace oneflow {

IDMgr::IDMgr() {
  regst_desc_id_count_ = 0;
  mem_block_id_count_ = 0;
  chunk_id_count_ = 0;
  char* id_count_start = std::getenv("ID_START");
  if (id_count_start) {
    auto id_start = std::stoi(id_count_start);
    regst_desc_id_count_ = id_start;
    mem_block_id_count_ = id_start;
    chunk_id_count_ = id_start;
  }
}

}  // namespace oneflow
