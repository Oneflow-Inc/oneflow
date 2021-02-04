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

namespace oneflow {

namespace one {

class Device;

class TensorImpl {
 public:
  TensorImpl() = default;
  // Tensor_Mirrored 模式构造函数
  TensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
             const std::shared_ptr<Device>& device)
             : shape_(shape), dtype_(dtype), device_(device) {}
  
  // Tensor_Consistent 模式构造函数
  TensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
             const std::shared_ptr<compatible_py::Distribute>&distribute,
             const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
             : shape_(shape), dtype_(dtype), distribute_(distribute), parallel_conf_(parallel_conf) {}
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
  virtual std::string job_name() { UNIMPLEMENTED(); }
  virtual void set_job_name(std::string job_name) { UNIMPLEMENTED(); }
  virtual int64_t numpy_size() const { UNIMPLEMENTED(); }
  virtual int64_t numpy_list_size() const { UNIMPLEMENTED(); }
  virtual std::shared_ptr<compatible_py::BlobObject> blob_object() const { UNIMPLEMENTED(); }
  virtual int64_t split_axis() const { UNIMPLEMENTED(); };
  virtual bool is_dynamic() const { UNIMPLEMENTED(); }
  virtual bool is_tensor_list() const { UNIMPLEMENTED(); }

 protected:
  Maybe<std::string> Distribute2Str() const;
  std::shared_ptr<cfg::LogicalBlobId> lbi_;
  std::shared_ptr<compatible_py::Distribute> distribute_;
  std::string lbn_;
  int64_t parallel_size_;
  std::string job_name_;
  std::shared_ptr<compatible_py::BlobObject>& blob_object_; //Blob_Eager 模式构造函数
  std::shared_ptr<compatible_py::BlobRegister>& blob_register_; //Blob_Eager 模式构造函数

 private:
  // Tensor_Mirrored 模式
  std::shared_ptr<Shape> shape_;
  DataType dtype_;
  std::shared_ptr<Device> device_;

  // Tensor_Consistent 模式
  std::shared_ptr<compatible_py::Distribute> distribute_;
  std::shared_ptr<cfg::ParallelConf> parallel_conf_;
};

/* class MirroredLazyTensorImpl:public LazyTensorImpl:public TensorImpl */
class LazyTensorImpl : public TensorImpl {
 public:
  // Mirrored 模式构造函数
  LazyTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                 const std::shared_ptr<Device>& device)
      : TensorImpl(shape, dtype, device) {}

  // Consistent 模式构造函数
  LazyTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                 const std::shared_ptr<compatible_py::Distribute>&distribute,
                 const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
      : TensorImpl(shape, dtype, distribute, parallel_conf) {}
  virtual ~LazyTensorImpl() = default;
};

class MirroredLazyTensorImpl : public LazyTensorImpl {
 public:
  MirroredLazyTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                         const std::shared_ptr<Device>& device)
      : LazyTensorImpl(shape, dtype, device) {}
  ~MirroredLazyTensorImpl() = default;
};

/* class ConsistentLazyTensorImpl:public LazyTensorImpl:public TensorImpl */
class ConsistentLazyTensorImpl : public LazyTensorImpl{
  public:
  ConsistentLazyTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                            const std::shared_ptr<compatible_py::Distribute>& distribute, 
                            const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
        : LazyTensorImpl(shape, dtype, distribute, parallel_conf){}
  ~ConsistentLazyTensorImpl() = default;
};

/* class MirroredLazyBlobDesc:public LazyBlobDesc:public TensorImpl */
class LazyBlobDesc : public TensorImpl {
 public:
  LazyBlobDesc(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                 const std::shared_ptr<compatible_py::Distribute>& distribute)
      : TensorImpl(lbi, job_name, distribute){}
  virtual ~LazyBlobDesc() = default;
};

class MirroredLazyBlobDesc : public LazyBlobDesc {
 public:
  MirroredLazyBlobDesc(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                       const std::shared_ptr<compatible_py::Distribute>& distribute)
      : LazyBlobDesc(lbi, job_name, distribute);
  ~MirroredLazyBlobDesc() = default;

  virtual std::string get_mirror_shape_log_warning() const;
  std::shared_ptr<Shape> shape() const override;
  DataType dtype() const override;
  int64_t batch_axis() const override;
  int64_t split_axis() const override;
  bool is_dynamic() const override;
  bool is_tensor_list() const override;
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override;
};

/* class ConsistentLazyBlobDesc:public LazyBlobDesc:public TensorImpl */
class ConsistentLazyBlobDesc:public LazyBlobDesc {
  public:
  ConsistentLazyBlobDesc(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                         const std::string& job_name,
                         const std::shared_ptr<compatible_py::Distribute>& distribute)
    : LazyBlobDesc(lbi, job_name, distribute){}
  ~ConsistentLazyBlobDesc()=default;

  virtual std::string get_lazy_shape_log_warning() const;    // ToDo 
  std::shared_ptr<Shape> shape() const override;
  DataType dtype() const override;
  int64_t batch_axis() const override;
  int64_t split_axis() const override;
  bool is_dynamic() const override;
  bool is_tensor_list() const override;
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override;
  bool IdenticalTo(const std::shared_ptr<ConsistentLazyBlobDesc>& rhs) const;
};

/* class MirroredEagerTensorImpl:public EagerTensorImpl:public TensorImpl */
class EagerTensorImpl : public TensorImpl {
 public:
  // Mirrored 模式构造函数
  EagerTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                  const std::shared_ptr<Device>& device)
      : TensorImpl(shape, dtype, device) {}

  //Consistent 模式构造函数
  EagerTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                  const std::shared_ptr<compatible_py::Distribute>& distribute, 
                  const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
      : TensorImpl(shape, dtype, distribute, parallel_conf) {}
  virtual ~EagerTensorImpl();
};

class MirroredEagerTensorImpl : public EagerTensorImpl {
 public:
  MirroredEagerTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                          const std::shared_ptr<Device>& device)
      : EagerTensorImpl(shape, dtype, device);
};

/* class ConsistentEagerTensorImpl:public EagerTensorImpl:public TensorImpl */
class ConsistentEagerTensorImpl : public EagerTensorImpl {
  public:
  ConsistentEagerTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                            const std::shared_ptr<compatible_py::Distribute>& distribute, 
                            const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
      :EagerTensorImpl(shape, dtype, distribute, parallel_conf);
};

/* class MirroredEagerBlobDesc:public EagerBlobDesc:public TensorImpl */
class EagerBlobDesc : public TensorImpl {
 public:
  EagerBlobDesc(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                const std::shared_ptr<compatible_py::Distribute>& distribute,
                const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                const std::shared_ptr<compatible_py::BlobRegister>& blob_register);
  virtual ~EagerBlobDesc();

  int64_t numpy_size() const override;
  int64_t numpy_list_size() const override;
  std::shared_ptr<Shape> shape() const override; // BlobHeaderTrait::shape()
  DataType dtype() const override;// BlobHeaderTrait::dtype()
  int64_t batch_axis() const override;
  int64_t split_axis() const override;
  bool is_dynamic() const override;
  bool is_tensor_list() const override;// BlobHeaderTrait::is_tensor_list()
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override;
  int64_t parallel_size() override;
  std::shared_ptr<compatible_py::BlobObject> blob_object() const override;
  void _Init(const std::shared_ptr<compatible_py::BlobObject>& blob_object,
             const std::shared_ptr<compatible_py::BlobRegister>& blob_register);
  bool IdenticalTo(const std::shared_ptr<EagerBlobDesc>& rhs) const;

 private:
  std::shared_ptr<compatible_py::RegisteredBlobAccess> registered_blob_access_;
};

class MirroredEagerBlobDesc : public EagerBlobDesc {
 public:
  MirroredEagerBlobDesc(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                        const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                        const std::shared_ptr<compatible_py::BlobRegister>& blob_register,
                        const std::string& job_name,
                        const std::shared_ptr<compatible_py::Distribute>& distribute);
  ~MirroredEagerBlobDesc() override = default;
};

/* class ConsistentEagerBlobDesc:public EagerBlobDesc:public TensorImpl */
class ConsistentEagerBlobDesc:public EagerBlobDesc{
  public:
  ConsistentEagerBlobDesc(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
  const std::shared_ptr<compatible_py::BlobObject>& blob_object,
  const std::shared_ptr<compatible_py::BlobRegister>& blob_register,
  const std::string& job_name,
  const std::shared_ptr<compatible_py::Distribute>& distribute);   // ToDo 
  ~ConsistentEagerBlobDesc() override = default;
};

}  // namespace one

}  // namespace oneflow
