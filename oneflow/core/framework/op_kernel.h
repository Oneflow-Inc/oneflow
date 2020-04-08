#ifndef ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_

#include <glog/logging.h>
#include "oneflow/core/framework/util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/placement.pb.h"

namespace oneflow {

namespace user_op {

class KernelInitContext {
 public:
  virtual ~KernelInitContext() = default;

  virtual DeviceCtx* device_ctx() = 0;

  virtual DeviceType device_type() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;
  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string&, int32_t) const = 0;

  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;

  template<typename T>
  T GetAttr(const std::string& attr_name) const {
    return user_op_conf_.attr<T>(attr_name);
  }

 protected:
  KernelInitContext(UserOpConfWrapper&& conf) : user_op_conf_(std::move(conf)) {}
  KernelInitContext(const KernelInitContext&) = delete;

 private:
  UserOpConfWrapper user_op_conf_;
};

class Tensor;

class KernelComputeContext {
 public:
  virtual ~KernelComputeContext() = default;

  virtual Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) = 0;
  virtual DeviceCtx* device_ctx() = 0;

  virtual DeviceType device_type() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;

  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;

  template<typename T>
  T GetAttr(const std::string& attr_name) const {
    return user_op_conf_.attr<T>(attr_name);
  }

 protected:
  KernelComputeContext(UserOpConfWrapper&& conf) : user_op_conf_(conf) {}
  KernelComputeContext(const KernelComputeContext&) = delete;

 private:
  UserOpConfWrapper user_op_conf_;
};

class OpKernelContext {
 public:
  virtual ~OpKernelContext() = default;

 protected:
  OpKernelContext() = default;
};

template<typename T>
class OpKernelContextIf final : public OpKernelContext {
 public:
  template<typename... Args>
  explicit OpKernelContextIf(Args&&... args) : data_(std::forward<Args>(args)...) {}

  ~OpKernelContextIf() = default;

  const T& Get() const { return data_; }
  T* Mutable() { return &data_; }

 private:
  T data_;
};

class OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernel);
  virtual ~OpKernel() = default;

  virtual void InitOpKernelContext(KernelInitContext* ctx, OpKernelContext**) const {}

  virtual void Compute(KernelComputeContext* ctx, OpKernelContext*) const { Compute(ctx); }
  virtual void Compute(KernelComputeContext*) const { LOG(INFO) << "UNIMPLEMENTED"; }

 protected:
  OpKernel() = default;
};

}  // namespace user_op

}  // namespace oneflow

#endif
