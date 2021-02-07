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
#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.cfg.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/framework/tensor_impl.h"

namespace oneflow {

class Blob;

namespace cfg {

class LogicalBlobId;
class ParallelConf;

}  // namespace cfg

class Tensor {
 public:
  virtual ~Tensor() = default;

  virtual std::shared_ptr<cfg::LogicalBlobId> lbi() const = 0;
  virtual std::string logical_blob_name() const = 0;
  virtual std::string op_name() const = 0;
  virtual std::string blob_name() const = 0;
  virtual std::shared_ptr<Shape> shape() const = 0;
  virtual DataType dtype() const = 0;
  virtual std::shared_ptr<cfg::ParallelConf> parallel_conf() const = 0;
};

namespace one {

class Device {
 public:
  Device(DeviceType device_type, int64_t device_id)
      : device_id_(device_id), device_type_(device_type) {}
  DeviceType device_type() const { return device_type_; }
  int64_t device_id() const { return device_id_; }

 private:
  int64_t device_id_;
  DeviceType device_type_;
};

class Tensor {
 public:
  virtual ~Tensor() = default;
  std::shared_ptr<Shape> shape() const { return impl_->shape(); }
  DataType dtype() const { return impl_->dtype(); }
  std::shared_ptr<Device> device() const { return impl_->device(); }
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const { return impl_->parallel_conf(); }
  bool is_lazy() const;

  std::shared_ptr<cfg::LogicalBlobId> lbi() const { return impl_->lbi(); }
  std::string logical_blob_name() const { return impl_->logical_blob_name(); }
  std::string op_name() const { return impl_->op_name(); }
  std::string blob_name() const { return impl_->blob_name(); }
  std::string unique_name() const { return impl_->unique_name(); }
  void set_distribute(const std::shared_ptr<compatible_py::Distribute> distribute) {
    impl_->set_distribute(distribute);
  }
  std::shared_ptr<compatible_py::Distribute> distribute() const { return impl_->distribute(); }
  bool has_batch_axis() const { return impl_->has_batch_axis(); }
  std::string job_name() const { return impl_->job_name(); }

  int64_t batch_axis() const { return impl_->batch_axis(); }
  int64_t parallel_size() { return impl_->parallel_size(); }
  void set_job_name(std::string job_name) { impl_->set_job_name(job_name); }
  int64_t numpy_size() const { return impl_->numpy_size(); }
  int64_t numpy_list_size() const { return impl_->numpy_list_size(); }
  std::shared_ptr<compatible_py::BlobObject> blob_object() const { return impl_->blob_object(); }
  int64_t split_axis() const { return impl_->split_axis(); }
  bool is_dynamic() const { return impl_->is_dynamic(); }
  bool is_tensor_list() const { return impl_->is_tensor_list(); }

  std::shared_ptr<Blob> storage() const { return impl_->storage(); }
  bool has_storage() const { return impl_->has_storage(); }

  template<typename T = void>
  const T* data() const { return impl_->data<T>(); }

  template<typename T = void>
  T* mutable_data() { return impl_->mutable_data<T>(); }

 protected:
  std::shared_ptr<TensorImpl> impl_;
};

class MirroredTensor : public Tensor {
 public:
  MirroredTensor(const std::shared_ptr<Shape>& shape, DataType dtype,
                 const std::shared_ptr<Device>& device);
  MirroredTensor(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                 const std::shared_ptr<compatible_py::Distribute>& distribute);
  MirroredTensor(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                 const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                 const std::shared_ptr<compatible_py::BlobRegister>& blob_register,
                 const std::string& job_name,
                 const std::shared_ptr<compatible_py::Distribute>& distribute);
  ~MirroredTensor() = default;
};

class ConsistentTensor : public Tensor {
 public:
  ConsistentTensor(const std::shared_ptr<Shape>& shape, DataType dtype,
                   const std::shared_ptr<compatible_py::Distribute>& distribute,
                   std::shared_ptr<cfg::ParallelConf>& parallel_conf);
  ConsistentTensor(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                   const std::shared_ptr<compatible_py::Distribute>& distribute);
  ConsistentTensor(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                   const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                   const std::shared_ptr<compatible_py::BlobRegister>& blob_register,
                   const std::string& job_name,
                   const std::shared_ptr<compatible_py::Distribute>& distribute);
  ~ConsistentTensor() = default;
};

}  // namespace one

namespace user_op {

class Tensor {
 public:
  ~Tensor() = default;

  virtual const ShapeView& shape() const = 0;
  virtual MutShapeView* mut_shape() = 0;
  virtual DataType data_type() const = 0;
  virtual const MemoryCase& mem_case() const = 0;
  virtual const void* raw_dptr() const = 0;
  virtual void* mut_raw_dptr() = 0;

  template<typename T = void>
  const T* dptr() const {
    CheckDataType<T>();
    return reinterpret_cast<const T*>(raw_dptr());
  }

  template<typename T = void>
  T* mut_dptr() {
    CheckDataType<T>();
    return reinterpret_cast<T*>(mut_raw_dptr());
  }

 protected:
  template<typename T>
  void CheckDataType() const {
    LOG_IF(FATAL, (std::is_same<T, void>::value == false && std::is_same<T, char>::value == false
                   && data_type() != DataType::kChar && data_type() != GetDataType<T>::value))
        << "tensor data_type mismatched. value: " << DataType_Name(data_type())
        << ", template T:" << DataType_Name(GetDataType<T>::value);
  }
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_H_
