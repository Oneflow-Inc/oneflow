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
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/memory/memory_allocator.h"

namespace oneflow {

namespace detail {

static constexpr double kDefaultGrowthFactor = 1.0f;
static constexpr double kDefaultShrinkFactor = 0.7f;
static constexpr size_t kDefaultTensorBufferAlignedSize = 1024;

size_t GetTensorBufferAlignedSize(size_t origin_size, double factor) {
  static size_t aligned_size =
      ParseIntegerFromEnv("ONEFLOW_TENSOR_BUFFER_ALIGNED_SIZE", kDefaultTensorBufferAlignedSize);
  return RoundUp(static_cast<size_t>(origin_size * factor), aligned_size);
}

size_t GetTensorBufferGrowthSize(size_t origin_size) {
  static double factor =
      ParseFloatFromEnv("ONEFLOW_TENSOR_BUFFER_GROWTH_FACTOR", kDefaultGrowthFactor);
  return GetTensorBufferAlignedSize(origin_size, factor);
}

size_t GetTensorBufferShrinkSize(size_t origin_size) {
  static double factor =
      ParseFloatFromEnv("ONEFLOW_TENSOR_BUFFER_SHRINK_FACTOR", kDefaultShrinkFactor);
  return GetTensorBufferAlignedSize(origin_size, factor);
}

void CheckTensorBufferDataType(DataType val) {
  CHECK(val != DataType::kTensorBuffer && val != DataType::kOFRecord)
      << "TensorBuffer only support POD as internal data type.";
}

void TensorBufferImpl::Reset(const Shape& shape, DataType dtype) {
  int64_t elem_cnt = shape.elem_cnt();
  if (dtype == DataType::kInvalidDataType || elem_cnt == 0) { return; }
  CheckTensorBufferDataType(dtype);

  if (shape == shape_ && dtype == data_type_) { return; }

  shape_ = shape;
  data_type_ = dtype;

  size_t new_buffer_size = elem_cnt * GetSizeOfDataType(dtype);
  Reserve(new_buffer_size);
}

void TensorBufferImpl::Reset(const Shape& shape) { Reset(shape, data_type_); }

void TensorBufferImpl::Reset(DataType dtype) {
  CheckTensorBufferDataType(dtype);
  if (dtype == DataType::kInvalidDataType) {
    Reset();
  } else {
    Reset(shape_, dtype);
  }
}

void TensorBufferImpl::Reset() {
  shape_ = Shape();
  data_type_ = DataType::kInvalidDataType;
  DeallocateBuffer();
}

void TensorBufferImpl::AllocateBuffer(size_t size) {
  CHECK(buffer_ == nullptr);
  buffer_ = MemoryAllocatorImpl::AllocateUnPinnedHostMem(size);
  buffer_size_ = size;
}

void TensorBufferImpl::DeallocateBuffer() {
  if (buffer_) { MemoryAllocatorImpl::DeallocateUnPinnedHostMem(buffer_); }
  buffer_ = nullptr;
  buffer_size_ = 0;
}

void TensorBufferImpl::Reserve(size_t new_size) {
  if (new_size > buffer_size_) {
    size_t growth_size = std::max(new_size, GetTensorBufferGrowthSize(new_size));
    DeallocateBuffer();
    AllocateBuffer(growth_size);
  } else {
    size_t shrink_size = GetTensorBufferShrinkSize(buffer_size_);
    if (new_size <= shrink_size) {
      DeallocateBuffer();
      AllocateBuffer(shrink_size);
    }
  }
}

void TensorBufferImpl::CopyFrom(const TensorBufferImpl* src) {
  if (src == this) { return; }
  Reset(src->shape(), src->data_type());
  memcpy(buffer_, src->buffer(), buffer_size_);
}

void TensorBufferImpl::Swap(TensorBufferImpl* other) {
  std::swap(buffer_, other->buffer_);
  std::swap(buffer_size_, other->buffer_size_);
  std::swap(shape_, other->shape_);
  std::swap(data_type_, other->data_type_);
}

}  // namespace detail

TensorBuffer::~TensorBuffer() {
  if (auto* pool = TensorBufferPool::TryGet()) { pool->Deallocate(&impl_); }
}

TensorBuffer::TensorBuffer(const Shape& shape, DataType dtype) { Allocate(shape, dtype); }

TensorBuffer& TensorBuffer::operator=(TensorBuffer&& other) noexcept {
  impl_ = std::move(other.impl_);
  return *this;
}

void TensorBuffer::Allocate(const Shape& shape, DataType dtype) {
  CHECK(!is_allocated());
  if (auto* pool = TensorBufferPool::TryGet()) {
    pool->Allocate(&impl_, shape, dtype);
  } else {
    impl_.reset(new detail::TensorBufferImpl(shape, dtype));
  }
}

void TensorBuffer::Reset(const Shape& shape, DataType dtype) {
  if (is_allocated()) {
    impl_->Reset(shape, dtype);
  } else {
    Allocate(shape, dtype);
  }
}

void TensorBuffer::Reset(const Shape& shape) {
  CHECK(is_allocated()) << "TensorBuffer is not allocated";
  impl_->Reset(shape);
}

void TensorBuffer::Reset(DataType dtype) {
  CHECK(is_allocated()) << "TensorBuffer is not allocated";
  impl_->Reset(dtype);
}

void TensorBuffer::Reset() {
  if (impl_) { impl_->Reset(); }
}

const Shape& TensorBuffer::shape() const {
  CHECK(is_allocated()) << "TensorBuffer is not allocated";
  return impl_->shape();
}

DataType TensorBuffer::data_type() const {
  CHECK(is_allocated()) << "TensorBuffer is not allocated";
  return impl_->data_type();
}

void* TensorBuffer::raw_data() {
  CHECK(is_allocated()) << "TensorBuffer is not allocated";
  return impl_->buffer();
}

const void* TensorBuffer::raw_data() const {
  CHECK(is_allocated()) << "TensorBuffer is not allocated";
  return const_cast<detail::TensorBufferImpl*>(impl_.get())->buffer();
}

void TensorBuffer::CopyFrom(const TensorBuffer& src) {
  CHECK(src.is_allocated()) << "TensorBuffer src is not allocated";
  if (!is_allocated()) { Allocate(src.shape(), src.data_type()); }
  impl_->CopyFrom(src.impl_.get());
}

void TensorBuffer::Swap(TensorBuffer& other) { std::swap(impl_, other.impl_); }

namespace {

constexpr size_t kDefaultPoolSizeBase = 64;
constexpr double kDefaultPoolSizeFactor = 2.0;
constexpr size_t kDefaultThreadLocalCacheSize = 64;

size_t GetTensorBufferPoolSize(size_t base = kDefaultPoolSizeBase) {
  static double factor =
      ParseFloatFromEnv("ONEFLOW_TENSOR_BUFFER_POOL_SIZE_FACTOR", kDefaultPoolSizeFactor);
  return static_cast<size_t>(std::ceil(base * factor));
}

size_t GetTensorBufferPoolThreadLocalCacheSize() {
  static size_t cache_size = ParseIntegerFromEnv(
      "ONEFLOW_TENSOR_BUFFER_POOL_THREAD_LOCAL_CACHE_SIZE", kDefaultThreadLocalCacheSize);
  return cache_size;
}

}  // namespace

TensorBufferPool::TensorBufferPool()
    : thread_local_cache_size_(GetTensorBufferPoolThreadLocalCacheSize()),
      pool_size_(GetTensorBufferPoolSize()) {
  auto& thread_local_cache = ThreadLocalCache();
  thread_local_cache.reserve(thread_local_cache_size_);
  global_free_list_.reserve(pool_size_);
}

void TensorBufferPool::Allocate(ItemT* item, const Shape& shape, DataType dtype) {
  CHECK(!(*item)) << "TensorBuffer is already allocated";
  auto& thread_local_cache = ThreadLocalCache();
  if (thread_local_cache.empty() && thread_local_cache_size_ > 0) {
    std::unique_lock<std::mutex> lck(mtx_);
    if (!global_free_list_.empty()) {
      // fetch half of thread_local_cache_size of tensor buffers from global free list
      size_t fetches = thread_local_cache_size_ / 2;
      auto begin = global_free_list_.size() >= fetches ? (global_free_list_.end() - fetches)
                                                       : global_free_list_.begin();
      for (auto it = begin; it < global_free_list_.end(); ++it) {
        thread_local_cache.push_back(std::move(*it));
      }
      global_free_list_.erase(begin, global_free_list_.end());
    }
  }

  if (thread_local_cache.empty()) {
    item->reset(new detail::TensorBufferImpl(shape, dtype));
  } else {
    *item = std::move(thread_local_cache.back());
    thread_local_cache.pop_back();
    (*item)->Reset(shape, dtype);
  }
}

void TensorBufferPool::Deallocate(ItemT* item) {
  if (!(*item)) { return; }
  auto& thread_local_cache = ThreadLocalCache();
  if (thread_local_cache.size() < thread_local_cache_size_) {
    thread_local_cache.push_back(std::move(*item));
  } else {
    size_t releases = thread_local_cache.size() / 2;
    {
      std::unique_lock<std::mutex> lck(mtx_);
      if (global_free_list_.size() < pool_size_) {
        global_free_list_.push_back(std::move(*item));
        // release half of tensor buffers in thread local cache back to global free list
        while (global_free_list_.size() < pool_size_ && releases > 0) {
          global_free_list_.push_back(std::move(thread_local_cache.back()));
          thread_local_cache.pop_back();
          releases--;
        }
      }
    }
    // global free list is also full, release half of thread local cache
    thread_local_cache.resize(thread_local_cache.size() - releases);
  }
  if (*item) { item->reset(); }
}

void TensorBufferPool::IncreasePoolSizeByBase(size_t base) {
  std::unique_lock<std::mutex> lck(mtx_);
  pool_size_ += GetTensorBufferPoolSize(base);
  if (pool_size_ > global_free_list_.capacity()) { global_free_list_.reserve(pool_size_); }
  if (pool_size_ < global_free_list_.size()) { global_free_list_.resize(pool_size_); }
}

void TensorBufferPool::DecreasePoolSizeByBase(size_t base) {
  std::unique_lock<std::mutex> lck(mtx_);
  size_t dec = GetTensorBufferPoolSize(base);
  CHECK_GE(pool_size_, dec) << "pool_size " << pool_size_ << " decreased by " << dec
                            << " would be negative";
  pool_size_ -= dec;
  if (pool_size_ > global_free_list_.capacity()) { global_free_list_.reserve(pool_size_); }
  if (pool_size_ < global_free_list_.size()) { global_free_list_.resize(pool_size_); }
}

}  // namespace oneflow
