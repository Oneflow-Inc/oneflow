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
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace one {

class Device;

class TensorImpl {
 public:
  TensorImpl() = default;
  TensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
             const std::shared_ptr<Device>& device)
      : shape_(shape), dtype_(dtype), device_(device) {}
  TensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
             const std::shared_ptr<compatible_py::Distribute>& distribute,
             const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
      : distribute_(distribute), shape_(shape), dtype_(dtype), parallel_conf_(parallel_conf) {}
  TensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
             const std::shared_ptr<compatible_py::Distribute>& distribute);
  virtual ~TensorImpl() = default;

  virtual std::shared_ptr<Shape> shape() const { return shape_; }
  virtual DataType dtype() const { return dtype_; }
  std::shared_ptr<Device> device() const { return device_; }

  std::shared_ptr<cfg::LogicalBlobId> lbi() const;
  std::string logical_blob_name() const;
  std::string op_name() const;
  std::string blob_name() const;
  std::string unique_name() const;
  void set_distribute(const std::shared_ptr<compatible_py::Distribute> distribute);
  std::shared_ptr<compatible_py::Distribute> distribute() const;
  bool has_batch_axis() const;
  std::string job_name() const { return job_name_; }

  virtual std::shared_ptr<cfg::ParallelConf> parallel_conf() const { UNIMPLEMENTED(); }

  virtual int64_t batch_axis() const { UNIMPLEMENTED(); }
  virtual int64_t parallel_size() { UNIMPLEMENTED(); }
  virtual void set_job_name(std::string job_name) { UNIMPLEMENTED(); }
  virtual int64_t numpy_size() const { UNIMPLEMENTED(); }
  virtual int64_t numpy_list_size() const { UNIMPLEMENTED(); }
  virtual std::shared_ptr<compatible_py::BlobObject> blob_object() const { UNIMPLEMENTED(); }
  virtual int64_t split_axis() const { UNIMPLEMENTED(); }
  virtual bool is_dynamic() const { UNIMPLEMENTED(); }
  virtual bool is_tensor_list() const { UNIMPLEMENTED(); }

  virtual std::shared_ptr<Blob> storage() const { UNIMPLEMENTED(); }
  virtual bool has_storage() const { UNIMPLEMENTED(); }

  template<typename T = void>
  const T* data() const {
    CHECK(has_storage());
    return storage()->dptr<T>();
  }

  template<typename T = void>
  T* mutable_data() {
    CHECK(has_storage());
    return storage()->mut_dptr<T>();
  }

 protected:
  Maybe<std::string> Distribute2Str() const;
  std::shared_ptr<cfg::LogicalBlobId> lbi_;
  std::shared_ptr<compatible_py::Distribute> distribute_;
  std::string lbn_;
  int64_t parallel_size_;
  std::string job_name_;

 private:
  std::shared_ptr<Shape> shape_;
  DataType dtype_;
  std::shared_ptr<Device> device_;
  std::shared_ptr<cfg::ParallelConf> parallel_conf_;
};

class LazyBlob : public TensorImpl {
 public:
  LazyBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
           const std::shared_ptr<compatible_py::Distribute>& distribute)
      : TensorImpl(lbi, job_name, distribute) {}
  virtual ~LazyBlob() = default;
};

class LazyTensorImpl : public TensorImpl {
 public:
  LazyTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                 const std::shared_ptr<Device>& device)
      : TensorImpl(shape, dtype, device) {}
  LazyTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                 const std::shared_ptr<compatible_py::Distribute>& distribute,
                 const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
      : TensorImpl(shape, dtype, distribute, parallel_conf) {}
  virtual ~LazyTensorImpl() = default;
};

class EagerBlob : public TensorImpl {
 public:
  EagerBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
            const std::shared_ptr<compatible_py::Distribute>& distribute);
  virtual ~EagerBlob();

  int64_t numpy_size() const override;
  int64_t numpy_list_size() const override;
  std::shared_ptr<Shape> shape() const override;
  DataType dtype() const override;
  int64_t batch_axis() const override;
  int64_t split_axis() const override;
  bool is_dynamic() const override;
  bool is_tensor_list() const override;
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override;
  int64_t parallel_size() override;
  std::shared_ptr<compatible_py::BlobObject> blob_object() const override;
  void _Init(const std::shared_ptr<compatible_py::BlobObject>& blob_object,
             const std::shared_ptr<compatible_py::BlobRegister>& blob_register);
  bool IdenticalTo(const std::shared_ptr<EagerBlob>& rhs) const;

 private:
  std::shared_ptr<compatible_py::RegisteredBlobAccess> registered_blob_access_;
};

class EagerTensorImpl : public TensorImpl {
 public:
  EagerTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                  const std::shared_ptr<Device>& device)
      : TensorImpl(shape, dtype, device) {}
  EagerTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                  const std::shared_ptr<compatible_py::Distribute>& distribute,
                  const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
      : TensorImpl(shape, dtype, distribute, parallel_conf) {}
  virtual ~EagerTensorImpl() = default;
};

class MirroredLazyTensorImpl : public LazyTensorImpl {
 public:
  MirroredLazyTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                         const std::shared_ptr<Device>& device)
      : LazyTensorImpl(shape, dtype, device) {}
  ~MirroredLazyTensorImpl() = default;
  std::shared_ptr<Blob> storage() const override { return storage_; }
  bool has_storage() const override { return static_cast<bool>(storage_); }

 private:
  std::shared_ptr<Blob> storage_;
};

class MirroredEagerTensorImpl : public EagerTensorImpl {
 public:
  MirroredEagerTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                          const std::shared_ptr<Device>& device)
      : EagerTensorImpl(shape, dtype, device) {}
  std::shared_ptr<Blob> storage() const override { return storage_; }
  bool has_storage() const override { return static_cast<bool>(storage_); }

 private:
  std::shared_ptr<Blob> storage_;
};

class ConsistentLazyTensorImpl : public LazyTensorImpl {
 public:
  ConsistentLazyTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                           const std::shared_ptr<compatible_py::Distribute>& distribute,
                           const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
      : LazyTensorImpl(shape, dtype, distribute, parallel_conf) {}
  ~ConsistentLazyTensorImpl() = default;
};

class ConsistentEagerTensorImpl : public EagerTensorImpl {
 public:
  ConsistentEagerTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                            const std::shared_ptr<compatible_py::Distribute>& distribute,
                            const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
      : EagerTensorImpl(shape, dtype, distribute, parallel_conf) {}
  ~ConsistentEagerTensorImpl() = default;
};

class MirroredLazyBlob : public LazyBlob {
 public:
  MirroredLazyBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                   const std::shared_ptr<compatible_py::Distribute>& distribute)
      : LazyBlob(lbi, job_name, distribute) {}
  ~MirroredLazyBlob() = default;

  virtual std::string get_mirror_shape_log_warning() const;
  std::shared_ptr<Shape> shape() const override;
  DataType dtype() const override;
  int64_t batch_axis() const override;
  int64_t split_axis() const override;
  bool is_dynamic() const override;
  bool is_tensor_list() const override;
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override;
};

class MirroredEagerBlob : public EagerBlob {
 public:
  MirroredEagerBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                    const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                    const std::shared_ptr<compatible_py::BlobRegister>& blob_register,
                    const std::string& job_name,
                    const std::shared_ptr<compatible_py::Distribute>& distribute);
  ~MirroredEagerBlob() override = default;
};

class ConsistentLazyBlob : public LazyBlob {
 public:
  ConsistentLazyBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                     const std::shared_ptr<compatible_py::Distribute>& distribute);
  ~ConsistentLazyBlob() = default;
  virtual std::string get_lazy_shape_log_warning() const;
  std::shared_ptr<Shape> shape() const override;
  DataType dtype() const override;
  int64_t batch_axis() const override;
  int64_t split_axis() const override;
  bool is_dynamic() const override;
  bool is_tensor_list() const override;
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override;
  bool IdenticalTo(const std::shared_ptr<ConsistentLazyBlob>& rhs) const;
};

class ConsistentEagerBlob : public EagerBlob {
 public:
  ConsistentEagerBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                      const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                      const std::shared_ptr<compatible_py::BlobRegister>& blob_register,
                      const std::string& job_name,
                      const std::shared_ptr<compatible_py::Distribute>& distribute);
  ~ConsistentEagerBlob() override = default;
};

}  // namespace one

}  // namespace oneflow
