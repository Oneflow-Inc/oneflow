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

static constexpr float kGrowthFactor = 1.0;
static constexpr float kShrinkThreshold = 0.9;
static constexpr size_t kTensorBufferAlignedSize = 1024;

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
  new_size = RoundUp(new_size, kTensorBufferAlignedSize);
  if (new_size > buffer_size_) {
    DeallocateBuffer();
    size_t growth_size = RoundUp(buffer_size_ * kGrowthFactor, kTensorBufferAlignedSize);
    new_size = std::max(new_size, growth_size);
    AllocateBuffer(new_size);
  } else {
    if (new_size < buffer_size_ * kShrinkThreshold) {
      DeallocateBuffer();
      AllocateBuffer(new_size);
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

TensorBuffer::~TensorBuffer() { TensorBufferPool::Get().Deallocate(*this); }

TensorBuffer::TensorBuffer(const Shape& shape, DataType dtype) {
  TensorBufferPool::Get().Allocate(*this, shape, dtype);
}

TensorBuffer& TensorBuffer::operator=(TensorBuffer&& other) noexcept {
  if (impl_) { TensorBufferPool::Get().Deallocate(*this); }
  impl_ = std::move(other.impl_);
  other.impl_.reset();
  return *this;
}

void TensorBuffer::Reset(const Shape& shape, DataType dtype) {
  if (is_allocated()) {
    impl_->Reset(shape, dtype);
  } else {
    TensorBufferPool::Get().Allocate(*this, shape, dtype);
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
  if (!is_allocated()) { TensorBufferPool::Get().Allocate(*this, src.shape(), src.data_type()); }
  impl_->CopyFrom(src.impl_.get());
}

void TensorBuffer::Swap(TensorBuffer& other) { std::swap(impl_, other.impl_); }

namespace {

constexpr size_t kDefaultPoolSize = 1024;
constexpr size_t kDefaultThreadLocalCacheSize = 64;

}  // namespace

TensorBufferPool::TensorBufferPool()
    : pool_size_(kDefaultPoolSize), thread_local_cache_size_(kDefaultThreadLocalCacheSize) {}

TensorBufferPool::~TensorBufferPool() {
  ThreadLocalCache().clear();
  std::unique_lock<std::mutex> lck(mtx_);
  global_free_list_.clear();
}

void TensorBufferPool::Allocate(TensorBuffer& tensor_buffer, const Shape& shape, DataType dtype) {
  CHECK(!tensor_buffer.impl_) << "TensorBuffer is already allocated";
  auto& thread_local_cache = ThreadLocalCache();
  if (thread_local_cache.empty() && thread_local_cache_size_ > 0) {
    thread_local_cache.reserve(thread_local_cache_size_);
    std::unique_lock<std::mutex> lck(mtx_);
    if (!global_free_list_.empty()) {
      auto begin = global_free_list_.size() >= thread_local_cache_size_
                       ? (global_free_list_.end() - thread_local_cache_size_)
                       : global_free_list_.begin();
      for (auto it = begin; it < global_free_list_.end(); ++it) {
        thread_local_cache.push_back(std::move(*it));
      }
      global_free_list_.erase(begin, global_free_list_.end());
    }
  }

  if (thread_local_cache.empty()) {
    tensor_buffer.impl_ = std::make_unique<detail::TensorBufferImpl>(shape, dtype);
  } else {
    tensor_buffer.impl_ = std::move(thread_local_cache.back());
    CHECK(tensor_buffer.impl_);
    thread_local_cache.pop_back();
    tensor_buffer.impl_->Reset(shape, dtype);
  }
}

void TensorBufferPool::Deallocate(TensorBuffer& tensor_buffer) {
  if (!tensor_buffer.impl_) { return; }
  std::unique_lock<std::mutex> lck(mtx_);
  if (global_free_list_.size() < pool_size_) {
    global_free_list_.push_back(std::move(tensor_buffer.impl_));
  }
  tensor_buffer.impl_.reset();
}

void TensorBufferPool::Deallocate(std::vector<TensorBuffer>& tensor_buffers) { UNIMPLEMENTED(); }

}  // namespace oneflow
