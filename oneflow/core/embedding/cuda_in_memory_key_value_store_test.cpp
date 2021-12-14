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
#include "oneflow/core/embedding/cuda_in_memory_key_value_store.h"
#include "oneflow/core/device/cuda_util.h"
#include <gtest/gtest.h>

namespace oneflow {

namespace embedding {

namespace {

#ifdef WITH_CUDA

TEST(CudaInMemoryKeyValueStore, PlainEncoder) {
  int device_count = 0;
  if (cudaGetDevice(&device_count) != cudaSuccess) { return; }
  if (device_count <= 0) { return; }

  CudaInMemoryKeyValueStoreOptions options{};
  options.num_shards = 4;
  options.embedding_vec_size = 128;
  options.num_embeddings = 1024 * 4;
  options.num_device_embeddings = 1024;
  options.encoding_type = CudaInMemoryKeyValueStoreOptions::EncodingType::kPlain;
  std::unique_ptr<KeyValueStore> store = NewCudaInMemoryKeyValueStore(options);
}

#endif  // WITH_CUDA

}  // namespace

}  // namespace embedding

}  // namespace oneflow
