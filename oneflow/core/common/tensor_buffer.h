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
#include "oneflow/core/common/shape_view.h"
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

  TensorBuffer(TensorBuffer&& other) noexcept : impl_(std::move(other.impl_)) {}
  TensorBuffer& operator=(TensorBuffer&& other) noexcept;

  bool is_allocated() const { return bool(impl_); }
  const Shape& shape() const;
  ShapeView shape_view() const { return shape(); }
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

  void Allocate(const Shape& shape, DataType dtype);
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
  using ItemT = std::unique_ptr<detail::TensorBufferImpl>;
  using ListT = std::vector<ItemT>;

  static TensorBufferPool* Get() {
    auto& ptr = GetPtr();
    CHECK(ptr) << "TensorBufferPool has not been created";
    return ptr.get();
  }

  static TensorBufferPool* TryGet() {
    auto& ptr = GetPtr();
    return ptr.get();
  }

  static void New() {
    auto& ptr = GetPtr();
    CHECK(!ptr) << "TensorBufferPool is already New";
    ptr.reset(new TensorBufferPool());
  }

  static void Delete() {
    auto& ptr = GetPtr();
    if (ptr) { ptr.reset(); }
  }

  ~TensorBufferPool() = default;
  OF_DISALLOW_COPY_AND_MOVE(TensorBufferPool);

  void Allocate(ItemT* item, const Shape& shape, DataType dtype);
  void Deallocate(ItemT* item);

  void IncreasePoolSizeByBase(size_t base);
  void DecreasePoolSizeByBase(size_t base);

 private:
  static std::unique_ptr<TensorBufferPool>& GetPtr() {
    static std::unique_ptr<TensorBufferPool> ptr;
    return ptr;
  }

  static ListT& ThreadLocalCache() {
    thread_local ListT thread_local_cache;
    return thread_local_cache;
  }

  TensorBufferPool();

  size_t thread_local_cache_size_;
  size_t pool_size_;

  ListT global_free_list_;
  std::mutex mtx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_TENSOR_BUFFER_H_
