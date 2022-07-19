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

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>
// #include "oneflow/api/cpp/framework/dtype.h"
// #include "oneflow/api/cpp/framework/shape.h"
#include "oneflow/api/cpp/tests/api_test.h"
#include "oneflow/api/cpp/one_embedding/one_embedding.h"

namespace oneflow_api {

namespace {

one_embedding::OneEmbeddingHandler LoadOneEmbeddingHandler() {
  std::string option("{\"name\": \"sparse_embedding\", \"key_type_size\": 8, \"value_type_size\": 4, \"value_type\": \"oneflow.float32\", \"storage_dim\": 128, \"kv_store\": {\"caches\": [{\"policy\": \"full\", \"capacity\": 33762577, \"value_memory_kind\": \"device\"}], \"persistent_table\": {\"path\": \"/home/zhengzekang/models/RecommenderSystems/dlrm/init_model\", \"physical_block_size\": 512, \"capacity_hint\": 33762577}}, \"parallel_num\": 1}"); 
  one_embedding::OneEmbeddingHandler handler(option);  
  handler.LoadSnapeshot("2022-07-18-21-45-00-867156")
  return handler;
}

}  // namespace

#ifdef WITH_CUDA
TEST(Api, graph_load_handler) {
  EnvScope scope;
  one_embedding::OneEmbeddingHandler handler = LoadOneEmbeddingHandler();
}

#endif

}  // namespace oneflow_api
