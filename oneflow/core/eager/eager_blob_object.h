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
#ifndef ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_
#define ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/op_args_reserved_size.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/tensor_methods.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/common/tensor_desc.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace one {

class LocalTensorMeta;
class MutLocalTensorMeta;

}  // namespace one

namespace vm {

class EagerBlobObject final : public user_op::Tensor,
                              public user_op::TensorDesc,
                              public std::enable_shared_from_this<EagerBlobObject> {
 public:
  EagerBlobObject(const EagerBlobObject&) = delete;
  EagerBlobObject(EagerBlobObject&&) = delete;
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case,
                  const Symbol<one::LocalTensorMeta>& static_local_tensor_meta,
                  const std::shared_ptr<const one::MutLocalTensorMeta>& dynamic_local_tensor_meta,
                  DataType data_type, const std::shared_ptr<TensorStorage>& tensor_storage)
      : EagerBlobObject(mem_case, static_local_tensor_meta, dynamic_local_tensor_meta, data_type,
                        tensor_storage, intrusive::shared_ptr<LocalDepObject>()) {}
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case,
                  const Symbol<one::LocalTensorMeta>& static_local_tensor_meta,
                  const std::shared_ptr<const one::MutLocalTensorMeta>& dynamic_local_tensor_meta,
                  DataType data_type, const std::shared_ptr<TensorStorage>& tensor_storage,
                  const intrusive::shared_ptr<LocalDepObject>& dep_object);

  ~EagerBlobObject() { tensor_storage_.reset(); }

  const std::shared_ptr<const one::MutLocalTensorMeta>& mut_tensor_meta() {
    return dynamic_local_tensor_meta_;
  }
  // Getters
  const Symbol<one::LocalTensorMeta>& tensor_meta() const { return static_local_tensor_meta_; }

  // user_op::TensorDesc overrides
  const Shape& shape() const override;
  const Stride& stride() const override;
  DataType data_type() const override { return data_type_; }
  bool is_dynamic() const override { return is_dynamic_; }

  void set_shape(const Shape& shape) override;
  void set_stride(const Stride& stride) override;
  void set_data_type(DataType data_type) override { data_type_ = data_type; }
  void set_is_dynamic(bool is_dynamic) override { is_dynamic_ = is_dynamic; }

  // user_op::Tensor overrides
  ShapeView shape_view() const override { return shape(); }
  MutShapeView mut_shape_view() override;
  const MemoryCase& mem_case() const override { return *mem_case_; }
  const void* raw_dptr() const override;
  void* mut_raw_dptr() override { return const_cast<void*>(raw_dptr()); }

  void set_storage_offset(const int64_t offset);

  // Returns true if allocate successfully.
  Maybe<bool> TryAllocateBlobBodyMemory(vm::Allocator* allocator);
  Maybe<void> DeallocateBlobDataPtr();
  void RegisterStorageDeleteHook(const std::function<void()>& hook);

  Maybe<LocalDepObject*> compute_local_dep_object() const {
    CHECK_NOTNULL_OR_RETURN(compute_local_dep_object_.get());
    return compute_local_dep_object_.get();
  }

  std::shared_ptr<TensorStorage>& tensor_storage() { return tensor_storage_; }

  const Optional<Symbol<::oneflow::Stream>>& producer_stream() const;
  Maybe<void> init_producer_stream(Symbol<::oneflow::Stream> producer_stream);

  const Optional<Symbol<::oneflow::Stream>>& last_used_stream() const;
  void set_last_used_stream(Symbol<::oneflow::Stream> last_used_stream);

  std::shared_ptr<const Shape> shape_ptr() const;
  std::shared_ptr<const Stride> stride_ptr() const;

  size_t ByteSizeOfBlobBody() const {
    const size_t elem_cnt = shape().elem_cnt();
    if (elem_cnt == 0) { return 0; }
    size_t max_offset = 0;
    for (size_t i = 0; i < shape().NumAxes(); ++i) {
      max_offset += (shape().at(i) - 1) * stride().at(i);
    }
    size_t capacity = max_offset + 1;
    // TODO(liujuncheng): remove this
    capacity = std::max<size_t>(capacity, elem_cnt);
    return capacity * GetSizeOfDataType(data_type_);
  }
  size_t AlignedByteSizeOfBlobBody() const {
    return RoundUp(ByteSizeOfBlobBody(), kBlobBodyAlignSize);
  }
  size_t ByteSizeOfBlobHeader() const { return shape().NumAxes() * sizeof(int64_t); }
  size_t AlignedByteSizeOfBlobHeader() const {
    return RoundUp(ByteSizeOfBlobHeader(), kBlobHeaderAlignSize);
  }

  const char* header_ptr() const { return reinterpret_cast<const char*>(shape().dim_vec().data()); }
  char* mut_header_ptr() {
    return reinterpret_cast<char*>(const_cast<int64_t*>(shape().dim_vec().data()));
  }

 private:
  bool is_dynamic_;
  std::shared_ptr<MemoryCase> mem_case_;
  DataType data_type_;
  int64_t storage_offset_;
  std::shared_ptr<TensorStorage> tensor_storage_;
  intrusive::shared_ptr<LocalDepObject> compute_local_dep_object_;

  Symbol<one::LocalTensorMeta> static_local_tensor_meta_;
  std::shared_ptr<const one::MutLocalTensorMeta> dynamic_local_tensor_meta_;
};

using EagerBlobObjectList = small_vector<std::shared_ptr<vm::EagerBlobObject>>;
using EagerBlobObjectListPtr = std::shared_ptr<const EagerBlobObjectList>;

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_
