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
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace embedding {

namespace {

class IteratorImpl : public KVBaseIterator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IteratorImpl);
  explicit IteratorImpl(Cache* cache)
      : cache_(cache),
        next_key_index_(0),
        max_key_index_(cache_->Capacity()),
        host_n_result_(nullptr),
        device_index_{} {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    OF_CUDA_CHECK(cudaMallocHost(&host_n_result_, sizeof(uint32_t)));
  }
  ~IteratorImpl() override {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFreeHost(host_n_result_));
  }

  void NextN(ep::Stream* stream, uint32_t n_request, uint32_t* n_result, void* keys,
             void* values) override {
    auto cuda_stream = stream->As<ep::CudaStream>();
    OF_CUDA_CHECK(cudaMemsetAsync(n_result, 0, sizeof(uint32_t), cuda_stream->cuda_stream()));
    while (true) {
      if (next_key_index_ >= max_key_index_) { return; }
      uint64_t end_key_index = next_key_index_ + n_request;
      if (end_key_index > max_key_index_) { end_key_index = max_key_index_; }
      cache_->Dump(stream, next_key_index_, end_key_index, n_result, keys, values);
      next_key_index_ = end_key_index;
      OF_CUDA_CHECK(cudaMemcpyAsync(host_n_result_, n_result, sizeof(uint32_t), cudaMemcpyDefault,
                                    cuda_stream->cuda_stream()));
      if (host_n_result_ != 0) { return; }
    }
  }

 private:
  Cache* cache_;
  uint64_t next_key_index_;
  uint64_t max_key_index_;
  uint32_t* host_n_result_;
  int device_index_;
};

}  // namespace

void Cache::WithIterator(const std::function<void(KVBaseIterator*)>& fn) {
  IteratorImpl iterator(this);
  fn(&iterator);
}

std::unique_ptr<Cache> NewCache(const CacheOptions& options) {
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
}

}  // namespace embedding

}  // namespace oneflow
