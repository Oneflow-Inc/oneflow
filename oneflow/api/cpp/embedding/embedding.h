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
#ifndef ONEFLOW_API_CPP_ONE_EMBEDDING_ONE_EMBEDDING_H_
#define ONEFLOW_API_CPP_ONE_EMBEDDING_ONE_EMBEDDING_H_

#include <string>

namespace oneflow_api {
namespace embedding {

// CreateKeyValueStore returns embedding name in the options.
std::string CreateKeyValueStore(const std::string& key_value_store_options, int64_t local_rank_id,
                                int64_t rank_id,
                                int64_t world_size);  // key_value_store_options is
                                                      // a serialized json string.
void LoadSnapshot(const std::string& snapshot_name, const std::string& embedding_name,
                  int64_t local_rank_id, int64_t rank_id);

}  // namespace embedding
}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_ONE_EMBEDDING_ONE_EMBEDDING_H_
