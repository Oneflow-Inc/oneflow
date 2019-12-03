#ifndef ONEFLOW_CORE_FRAMEWORK_KERNEL_CONTEXT_H_
#define ONEFLOW_CORE_FRAMEWORK_KERNEL_CONTEXT_H_

#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

class DeviceCtx;

namespace user_op {

class Blob;

class KernelContext {
 public:
  virtual ~KernelContext() = default;

  virtual Blob* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) = 0;
  virtual DeviceCtx* device_ctx() = 0;

  template<typename T>
  T GetAttr(const std::string& attr_name) const {
    return user_op_conf_.attr<T>(attr_name);
  }

 protected:
  KernelContext(UserOpConfWrapper&& conf) : user_op_conf_(conf) {}
  KernelContext(const KernelContext&) = delete;

 private:
  UserOpConfWrapper user_op_conf_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_KERNEL_CONTEXT_H_
