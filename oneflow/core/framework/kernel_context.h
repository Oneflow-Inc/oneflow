#ifndef ONEFLOW_CORE_FRAMEWORK_KERNEL_CONTEXT_H_
#define ONEFLOW_CORE_FRAMEWORK_KERNEL_CONTEXT_H_

#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/placement.pb.h"

namespace oneflow {

namespace user_op {

class Tensor;

class KernelContext {
 public:
  virtual ~KernelContext() = default;

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
  KernelContext(UserOpConfWrapper&& conf) : user_op_conf_(conf) {}
  KernelContext(const KernelContext&) = delete;

 private:
  UserOpConfWrapper user_op_conf_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_KERNEL_CONTEXT_H_
