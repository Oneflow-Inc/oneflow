#ifndef ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_STATE_WRAPPER_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_STATE_WRAPPER_H_

#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

template<typename T>
class OpKernelStateWrapper final : public user_op::OpKernelState {
 public:
  template<typename... Args>
  explicit OpKernelStateWrapper(Args&&... args) : data_(std::forward<Args>(args)...) {}

  ~OpKernelStateWrapper() = default;

  const T& Get() const { return data_; }
  T* Mutable() { return &data_; }

 private:
  T data_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_STATE_WRAPPER_H_
