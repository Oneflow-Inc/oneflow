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
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/framework/blob_register.h"
#include "oneflow/core/framework/py_remote_blob.h"

namespace oneflow {

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

class TensorImpl {
 public:
  TensorImpl() = default;
  virtual ~TensorImpl() = default;
  TensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
             const std::shared_ptr<Device>& device)
      : shape_(shape), dtype_(dtype), device_(device) {}
  TensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
             const std::shared_ptr<compatible_py::Distribute>& distribute,
             const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
      : shape_(shape), dtype_(dtype), parallel_conf_(parallel_conf), distribute_(distribute) {}

  virtual std::shared_ptr<Shape> shape() const { return shape_; }
  virtual void set_shape(const std::shared_ptr<Shape>& shape) { shape_ = shape; }
  virtual DataType dtype() const { return dtype_; }
  virtual void set_dtype(DataType dtype) { dtype_ = dtype; }
  virtual void set_parallel_conf(const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
    parallel_conf_ = parallel_conf;
  }
  virtual std::shared_ptr<Device> device() { return device_; }
  virtual void set_device(const std::shared_ptr<Device>& device) { device_ = device; }
  virtual void set_blob_desc(const std::shared_ptr<compatible_py::BlobDesc>& blob) { blob_ = blob; }
  virtual std::shared_ptr<cfg::ParallelConf> parallel_conf() const { UNIMPLEMENTED(); }
  virtual void set_distribute(const std::shared_ptr<compatible_py::Distribute> distribute) {
    blob_->set_distribute(distribute);
  }
  virtual std::shared_ptr<compatible_py::Distribute> distribute() const {
    return blob_->distribute();
  }
  virtual int64_t parallel_size() { UNIMPLEMENTED(); }
  virtual void set_job_name(std::string job_name) { UNIMPLEMENTED(); }
  virtual std::string job_name() const { UNIMPLEMENTED(); }
  virtual int64_t numpy_size() const { UNIMPLEMENTED(); }
  virtual int64_t numpy_list_size() const { UNIMPLEMENTED(); }
  virtual std::shared_ptr<compatible_py::BlobObject> blob_object() const { UNIMPLEMENTED(); }

  // deprecated in the future
  virtual std::shared_ptr<cfg::LogicalBlobId> lbi() const { return blob_->lbi(); }
  virtual std::string logical_blob_name() const { return blob_->logical_blob_name(); }
  virtual std::string op_name() const { return blob_->op_name(); }
  virtual std::string blob_name() const { return blob_->blob_name(); }
  virtual std::string unique_name() const { return blob_->unique_name(); }
  virtual bool has_batch_axis() const { return blob_->has_batch_axis(); }
  virtual int64_t batch_axis() const { return blob_->batch_axis(); }
  virtual int64_t split_axis() const { UNIMPLEMENTED(); }
  virtual bool is_dynamic() const { return blob_->is_dynamic(); }
  virtual bool is_tensor_list() const { return blob_->is_tensor_list(); }

 private:
  std::shared_ptr<Shape> shape_;
  DataType dtype_;
  std::shared_ptr<Device> device_;
  std::shared_ptr<cfg::ParallelConf> parallel_conf_;
  std::shared_ptr<compatible_py::Distribute> distribute_;
  std::shared_ptr<compatible_py::BlobDesc> blob_;
};

class Tensor {
 public:
  Tensor() = default;
  virtual ~Tensor() = default;
  std::shared_ptr<Shape> shape() const { return impl_->shape(); }
  void set_shape(const std::shared_ptr<Shape>& shape) { return impl_->set_shape(shape); }
  DataType dtype() const { return impl_->dtype(); }
  void set_dtype(DataType dtype) { return impl_->set_dtype(dtype); }
  std::shared_ptr<Device> device() const { return impl_->device(); }
  void set_device(const std::shared_ptr<Device>& device) { return impl_->set_device(device); }
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const { return impl_->parallel_conf(); }
  void set_parallel_conf(const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
    impl_->set_parallel_conf(parallel_conf);
  }
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
  void set_blob_desc(const std::shared_ptr<compatible_py::BlobDesc>& blob) {
    impl_->set_blob_desc(blob);
  }

 protected:
  std::shared_ptr<TensorImpl> impl_;
};

class MirroredTensor : public Tensor {
 public:
  MirroredTensor() = default;
  MirroredTensor(const std::shared_ptr<Shape>& shape, DataType dtype,
                 const std::shared_ptr<Device>& device);
  ~MirroredTensor() = default;
};

class ConsistentTensor : public Tensor {
 public:
  ConsistentTensor() = default;
  ConsistentTensor(const std::shared_ptr<Shape>& shape, DataType dtype,
                   const std::shared_ptr<compatible_py::Distribute>& distribute,
                   std::shared_ptr<cfg::ParallelConf>& parallel_conf);
  ~ConsistentTensor() = default;
};

class LazyTensorImpl : public TensorImpl {
 public:
  LazyTensorImpl() = default;
  virtual ~LazyTensorImpl() = default;
};

class EagerTensorImpl : public TensorImpl {
 public:
  EagerTensorImpl() = default;
  virtual ~EagerTensorImpl() = default;
};

class MirroredLazyTensorImpl : public LazyTensorImpl {
 public:
  MirroredLazyTensorImpl() = default;
  ~MirroredLazyTensorImpl();
};

class MirroredEagerTensorImpl : public EagerTensorImpl {
 public:
  MirroredEagerTensorImpl() {}
  ~MirroredEagerTensorImpl();
};

class ConsistentLazyTensorImpl : public LazyTensorImpl {
 public:
  ConsistentLazyTensorImpl() {}
  ~ConsistentLazyTensorImpl();
};

class ConsistentEagerTensorImpl : public EagerTensorImpl {
 public:
  ConsistentEagerTensorImpl() {}
  ~ConsistentEagerTensorImpl();
};

}  // namespace one

}  // namespace oneflow

