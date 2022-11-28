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
#include "oneflow/core/embedding/cache.h"
#include "oneflow/core/embedding/full_cache.h"
#include "oneflow/core/embedding/lru_cache.h"

namespace oneflow {

namespace embedding {

std::unique_ptr<Cache> NewCache(const CacheOptions& options) {
#ifdef WITH_CUDA
  CHECK_GT(options.key_size, 0);
  CHECK_GT(options.value_size, 0);
  CHECK_GT(options.capacity, 0);
  if (options.policy == CacheOptions::Policy::kLRU) {
    return NewLruCache(options);
  } else if (options.policy == CacheOptions::Policy::kFull) {
    return NewFullCache(options);
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
#else
  UNIMPLEMENTED();
  return nullptr;
#endif  // WITH_CUDA
}

}  // namespace embedding

}  // namespace oneflow
