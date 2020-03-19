#ifndef ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_

#include "oneflow/core/framework/util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/kernel_context.h"

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

class OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernel);
  virtual ~OpKernel() = default;

  virtual void Compute(KernelContext*) = 0;

 protected:
  OpKernel(KernelInitContext*) {}

 private:
};

}  // namespace user_op

}  // namespace oneflow

#endif
