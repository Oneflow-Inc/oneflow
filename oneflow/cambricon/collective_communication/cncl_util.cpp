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
#include "oneflow/cambricon/collective_communication/cncl_util.h"

namespace oneflow {

std::string CnclCliqueIdToString(const cnclCliqueId& unique_id) {
  char data[CNCL_CLIQUE_ID_BYTES_SIZE + sizeof(uint64_t)];
  memcpy(data, unique_id.data, CNCL_CLIQUE_ID_BYTES_SIZE);
  memcpy(data + CNCL_CLIQUE_ID_BYTES_SIZE, &unique_id.hash, sizeof(uint64_t));
  return std::string(data, CNCL_CLIQUE_ID_BYTES_SIZE + sizeof(uint64_t));
}

void CnclCliqueIdFromString(const std::string& str, cnclCliqueId* unique_id) {
  CHECK_EQ(str.size(), CNCL_CLIQUE_ID_BYTES_SIZE + sizeof(uint64_t));
  memcpy(unique_id->data, str.data(), CNCL_CLIQUE_ID_BYTES_SIZE);
  memcpy(&unique_id->hash, str.data() + CNCL_CLIQUE_ID_BYTES_SIZE, sizeof(uint64_t));
}

}  // namespace oneflow
