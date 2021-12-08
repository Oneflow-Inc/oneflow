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

class TensorBufferImpl final {
 public:
  TensorBufferImpl(const Shape& shape, DataType dtype) { Reset(shape, dtype); }
  ~TensorBufferImpl() = default;
  OF_DISALLOW_COPY_AND_MOVE(TensorBufferImpl);

  void Reset(const Shape& shape, DataType dtype);
  void Reset(const Shape& shape);
  void Reset(DataType dtype);
  void Reset();

  void CopyFrom(const TensorBufferImpl* src);
  void Swap(TensorBufferImpl* other);

  const Shape& shape() const { return shape_; }
  DataType data_type() const { return data_type_; }
  void* buffer() { return buffer_.get(); }
  const void* buffer() const { return buffer_.get(); }

 private:
  struct Deleter {
    void operator()(void* ptr) { MemoryAllocatorImpl::DeallocateUnPinnedHostMem(ptr); }
  };

  void Reserve(size_t new_size) {
    new_size = RoundUp(new_size, kTensorBufferAlignedSize);
    if (new_size > buffer_size_) {
      size_t growth_size = RoundUp(buffer_size_ * kGrowthFactor, kTensorBufferAlignedSize);
      new_size = std::max(new_size, growth_size);
      if (buffer_) { buffer_.reset(); }
    } else {
      if (new_size < buffer_size_ * kShrinkThreshold) { buffer_.reset(); }
    }
    buffer_.reset(MemoryAllocatorImpl::AllocateUnPinnedHostMem(new_size));
    buffer_size_ = new_size;
  }

  Shape shape_;
  DataType data_type_;

  std::unique_ptr<void, Deleter> buffer_;
  size_t buffer_size_;
};

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
  buffer_.reset();
  buffer_size_ = 0;
}

void TensorBufferImpl::CopyFrom(const TensorBufferImpl* src) {
  if (src == this) { return; }
  Reset(src->shape(), src->data_type());
  memcpy(buffer_.get(), src->buffer(), buffer_size_);
}

void TensorBufferImpl::Swap(TensorBufferImpl* other) {
  buffer_.swap(other->buffer_);
  std::swap(buffer_size_, other->buffer_size_);
  std::swap(shape_, other->shape_);
  std::swap(data_type_, other->data_type_);
}

}  // namespace detail

TensorBuffer::~TensorBuffer() { TensorBufferPool::Get().Deallocate(*this); }

TensorBuffer::TensorBuffer(const Shape& shape, DataType dtype) {
  TensorBufferPool::Get().Allocate(*this, shape, dtype);
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
  return const_cast<detail::TensorBufferImpl*>(impl_)->buffer();
}

void TensorBuffer::CopyFrom(const TensorBuffer& src) {
  CHECK(src.is_allocated()) << "TensorBuffer src is not allocated";
  if (!is_allocated()) { TensorBufferPool::Get().Allocate(*this, src.shape(), src.data_type()); }
  impl_->CopyFrom(src.impl_);
}

void TensorBuffer::Swap(TensorBuffer& other) { std::swap(impl_, other.impl_); }

TensorBufferPool::~TensorBufferPool() {
  std::unique_lock<std::mutex> lck(mtx_);
  for (auto* item : global_free_list_) { delete item; }
}

void TensorBufferPool::Allocate(TensorBuffer& tensor_buffer, const Shape& shape, DataType dtype) {
  if (tensor_buffer.impl_) {
    LOG(ERROR) << "TensorBuffer is already allocated";
    return;
  }

  thread_local TensorBufferList thread_local_cache;
  if (thread_local_cache.empty() && thread_local_cache_size_ > 0) {
    std::unique_lock<std::mutex> lck(mtx_);
    if (!global_free_list_.empty()) {
      auto begin = global_free_list_.size() >= thread_local_cache_size_
                       ? (global_free_list_.end() - thread_local_cache_size_)
                       : global_free_list_.begin();
      thread_local_cache.insert(thread_local_cache.end(), begin, global_free_list_.end());
    }
  }

  if (thread_local_cache.empty()) {
    tensor_buffer.impl_ = new detail::TensorBufferImpl(shape, dtype);
  } else {
    tensor_buffer.impl_ = thread_local_cache.back();
    thread_local_cache.pop_back();
  }
}

void TensorBufferPool::Deallocate(TensorBuffer& tensor_buffer) {
  if (!tensor_buffer.impl_) { return; }
  std::unique_lock<std::mutex> lck(mtx_);
  if (global_free_list_.size() < pool_size_) {
    global_free_list_.push_back(tensor_buffer.impl_);
    tensor_buffer.impl_ = nullptr;
  } else {
    delete tensor_buffer.impl_;
  }
}

}  // namespace oneflow
