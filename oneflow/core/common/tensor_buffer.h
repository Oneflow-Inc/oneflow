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
#ifndef ONEFLOW_CORE_COMMON_TENSOR_BUFFER_H_
#define ONEFLOW_CORE_COMMON_TENSOR_BUFFER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace detail {

class TensorBufferImpl final {
 public:
  TensorBufferImpl()
      : shape_(Shape()),
        data_type_(DataType::kInvalidDataType),
        buffer_(nullptr),
        buffer_size_(0) {}
  TensorBufferImpl(const Shape& shape, DataType dtype)
      : shape_(Shape()), data_type_(DataType::kInvalidDataType), buffer_(nullptr), buffer_size_(0) {
    Reset(shape, dtype);
  }
  ~TensorBufferImpl() { DeallocateBuffer(); }
  OF_DISALLOW_COPY_AND_MOVE(TensorBufferImpl);

  void Reset(const Shape& shape, DataType dtype);
  void Reset(const Shape& shape);
  void Reset(DataType dtype);
  void Reset();

  void CopyFrom(const TensorBufferImpl* src);
  void Swap(TensorBufferImpl* other);

  const Shape& shape() const { return shape_; }
  DataType data_type() const { return data_type_; }

  void* buffer() { return buffer_; }
  const void* buffer() const { return buffer_; }
  size_t buffer_size() const { return buffer_size_; }

 private:
  void AllocateBuffer(size_t size);
  void DeallocateBuffer();
  void Reserve(size_t new_size);

  Shape shape_;
  DataType data_type_;

  void* buffer_;
  size_t buffer_size_;
};

}  // namespace detail

class TensorBuffer final {
 public:
  TensorBuffer() = default;
  ~TensorBuffer();

  TensorBuffer(const Shape& shape, DataType dtype);

  TensorBuffer(const TensorBuffer&) = delete;
  TensorBuffer& operator=(const TensorBuffer&) = delete;

  TensorBuffer(TensorBuffer&& other) noexcept : impl_(std::move(other.impl_)) {
    other.impl_.reset();
  }
  TensorBuffer& operator=(TensorBuffer&& other) noexcept;

  bool is_allocated() const { return bool(impl_); }
  const Shape& shape() const;
  DataType data_type() const;
  int64_t elem_cnt() const { return shape().elem_cnt(); }
  size_t nbytes() const { return elem_cnt() * GetSizeOfDataType(data_type()); }

  void Reset(const Shape& shape, DataType dtype);
  void Reset(const Shape& shape);
  void Reset(DataType dtype);
  void Reset();

  // backward compatible interface and will be deprecated in future
  void Resize(const Shape& shape, DataType dtype) { Reset(shape, dtype); }

  void CopyFrom(const TensorBuffer& src);
  void Swap(TensorBuffer& other);

  template<typename T = void>
  T* mut_data() {
    if (raw_data() == nullptr) { return nullptr; }
    CheckDataType<T>(data_type());
    return static_cast<T*>(raw_data());
  }

  template<typename T = void>
  const T* data() const {
    if (raw_data() == nullptr) { return nullptr; }
    CheckDataType<T>(data_type());
    return static_cast<const T*>(raw_data());
  }

 private:
  friend class TensorBufferPool;

  void* raw_data();
  const void* raw_data() const;

  std::unique_ptr<detail::TensorBufferImpl> impl_;
};

#define BUFFER_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(TensorBuffer, DataType::kTensorBuffer)

template<>
struct GetDataType<TensorBuffer> : std::integral_constant<DataType, DataType::kTensorBuffer> {};
inline TensorBuffer GetTypeByDataType(std::integral_constant<DataType, DataType::kTensorBuffer>) {
  return {};
}

class TensorBufferPool final {
 public:
  using TensorBufferList = std::vector<std::unique_ptr<detail::TensorBufferImpl>>;

  static TensorBufferPool& Get() {
    if (!Ptr()) { Ptr().reset(new TensorBufferPool()); }
    return *Ptr().get();
  }

  static void Delete() {
    if (Ptr()) { Ptr().reset(); }
  }

  ~TensorBufferPool();
  OF_DISALLOW_COPY_AND_MOVE(TensorBufferPool);

  void Allocate(TensorBuffer& tensor_buffer, const Shape& shape, DataType dtype);
  void Deallocate(TensorBuffer& tensor_buffer);
  void Deallocate(std::vector<TensorBuffer>& tensor_buffers);

  void set_pool_size(size_t pool_size);
  void set_pool_size_base(size_t base);
  void set_thread_local_cache_size(size_t thread_local_cache_size);

 private:
  static std::unique_ptr<TensorBufferPool>& Ptr() {
    static std::unique_ptr<TensorBufferPool> ptr;
    return ptr;
  }

  static TensorBufferList& ThreadLocalCache() {
    thread_local TensorBufferList thread_local_cache;
    return thread_local_cache;
  }

  TensorBufferPool();

  size_t pool_size_;
  size_t thread_local_cache_size_;

  TensorBufferList global_free_list_;
  std::mutex mtx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_TENSOR_BUFFER_H_
