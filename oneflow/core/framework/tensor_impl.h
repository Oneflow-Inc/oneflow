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

#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_IMPL_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_IMPL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/framework/object.h"

namespace oneflow {

namespace compatible_py {
class Distribute;
}

namespace one {

class Device;

class TensorImpl {
 public:
  virtual ~TensorImpl() = default;
  virtual std::shared_ptr<Shape> shape() const = 0;
  virtual void set_shape(const std::shared_ptr<Shape>& shape) = 0;
  virtual DataType dtype() const = 0;
  virtual void set_dtype(DataType dtype) = 0;
  virtual void set_parallel_conf(const std::shared_ptr<cfg::ParallelConf>& parallel_conf)  = 0;
  virtual std::shared_ptr<cfg::ParallelConf> parallel_conf() const = 0;
};

class MirroredTensorImpl : public TensorImpl {
 public:
  virtual ~MirroredTensorImpl() = default;
  virtual std::shared_ptr<Device> device() const = 0;
  virtual void set_device(const std::shared_ptr<Device>& device) = 0;
};

class ConsistentTensorImpl : public TensorImpl {
 public:
  virtual ~ConsistentTensorImpl() = default;
  virtual std::shared_ptr<compatible_py::Distribute> distribute() const = 0;
  virtual void set_distribute(const std::shared_ptr<compatible_py::Distribute>& distribute) = 0;
};

class MirroredLazyTensorImpl : public MirroredTensorImpl {
 public:
  MirroredLazyTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                         const std::shared_ptr<Device>& device)
      : shape_(shape), dtype_(dtype), device_(device) {}
  ~MirroredLazyTensorImpl() = default;
  std::shared_ptr<Shape> shape() const override { return shape_; }
  void set_shape(const std::shared_ptr<Shape>& shape) override { shape_ = shape; }
  DataType dtype() const override { return dtype_; }
  void set_dtype(DataType dtype) override { dtype_ = dtype; }
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override { return parallel_conf_; }
  void set_parallel_conf(const std::shared_ptr<cfg::ParallelConf>& parallel_conf) override {
    parallel_conf_ = parallel_conf;
  }
  std::shared_ptr<Device> device() const override { return device_; }
  void set_device(const std::shared_ptr<Device>& device) override { device_ = device; }

 private:
  std::shared_ptr<Shape> shape_;
  DataType dtype_;
  std::shared_ptr<Device> device_;
  std::shared_ptr<cfg::ParallelConf> parallel_conf_;
};

class MirroredEagerTensorImpl : public MirroredTensorImpl {
 public:
  MirroredEagerTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                          const std::shared_ptr<Device>& device)
      : shape_(shape), dtype_(dtype), device_(device) {}
  ~MirroredEagerTensorImpl() = default;
  std::shared_ptr<Shape> shape() const override { return shape_; }
  void set_shape(const std::shared_ptr<Shape>& shape) override { shape_ = shape; }
  DataType dtype() const override { return dtype_; }
  void set_dtype(DataType dtype) override { dtype_ = dtype; }
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override { return parallel_conf_; }
  void set_parallel_conf(const std::shared_ptr<cfg::ParallelConf>& parallel_conf) override {
    parallel_conf_ = parallel_conf;
  }
  std::shared_ptr<Device> device() const override { return device_; }
  void set_blob_object(const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
    blob_object_ = blob_object;
  }
  std::shared_ptr<compatible_py::BlobObject> blob_object() const { return blob_object_; }
  void set_device(const std::shared_ptr<Device>& device) override { device_ = device; }

 private:
  std::shared_ptr<Shape> shape_;
  DataType dtype_;
  std::shared_ptr<Device> device_;
  std::shared_ptr<cfg::ParallelConf> parallel_conf_;
  std::shared_ptr<compatible_py::BlobObject> blob_object_;
};

class ConsistentLazyTensorImpl : public ConsistentTensorImpl {
 public:
  ConsistentLazyTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                           const std::shared_ptr<compatible_py::Distribute>& distribute,
                           const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
      : shape_(shape), dtype_(dtype), parallel_conf_(parallel_conf), distribute_(distribute) {}
  ~ConsistentLazyTensorImpl() = default;
  std::shared_ptr<Shape> shape() const override { return shape_; }
  void set_shape(const std::shared_ptr<Shape>& shape) override { shape_ = shape; }
  DataType dtype() const override { return dtype_; }
  void set_dtype(DataType dtype) override { dtype_ = dtype; }
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override { return parallel_conf_; }
  void set_parallel_conf(const std::shared_ptr<cfg::ParallelConf>& parallel_conf) override {
    parallel_conf_ = parallel_conf;
  }
  void set_distribute(const std::shared_ptr<compatible_py::Distribute>& distribute) override {
   distribute_ = distribute;
  }
  std::shared_ptr<compatible_py::Distribute> distribute() const override {
    return distribute_;
  }

 private:
  std::shared_ptr<Shape> shape_;
  DataType dtype_;
  std::shared_ptr<cfg::ParallelConf> parallel_conf_;
  std::shared_ptr<compatible_py::Distribute> distribute_;
};

class ConsistentEagerTensorImpl : public ConsistentTensorImpl {
 public:
  ConsistentEagerTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                            const std::shared_ptr<compatible_py::Distribute>& distribute,
                            const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
      : shape_(shape), dtype_(dtype), parallel_conf_(parallel_conf), distribute_(distribute) {}
  ~ConsistentEagerTensorImpl() = default;
  std::shared_ptr<Shape> shape() const override { return shape_; }
  void set_shape(const std::shared_ptr<Shape>& shape) override { shape_ = shape; }
  DataType dtype() const override { return dtype_; }
  void set_dtype(DataType dtype) override { dtype_ = dtype; }
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override { return parallel_conf_; }
  void set_parallel_conf(const std::shared_ptr<cfg::ParallelConf>& parallel_conf) override {
    parallel_conf_ = parallel_conf;
  }
  void set_distribute(const std::shared_ptr<compatible_py::Distribute>& distribute) override {
   distribute_ = distribute;
  }
  std::shared_ptr<compatible_py::Distribute> distribute() const override {
    return distribute_;
  }
  void set_blob_object(const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
    blob_object_ = blob_object;
  }
  std::shared_ptr<compatible_py::BlobObject> blob_object() const { return blob_object_; }

 private:
  std::shared_ptr<Shape> shape_;
  DataType dtype_;
  std::shared_ptr<cfg::ParallelConf> parallel_conf_;
  std::shared_ptr<compatible_py::Distribute> distribute_;
  std::shared_ptr<compatible_py::BlobObject> blob_object_;
};


}  // namespace one

}  // namespace oneflow
#endif

