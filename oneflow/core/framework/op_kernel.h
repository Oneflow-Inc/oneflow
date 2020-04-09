#ifndef ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_

#include <glog/logging.h>
#include <memory>
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

class OpKernelState {
 public:
  virtual ~OpKernelState() = default;

 protected:
  OpKernelState() = default;
};

class OpKernel;

template<typename T>
OpKernel* NewOpKernel();

enum OpKernelStatefulness {
  kInvalidOpKernelStatefulness = 0,
  kStatefulOpKernel,
  kStatelessOpKernel,
};

class OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernel);
  virtual ~OpKernel() = default;

  bool is_stateless() const {
    CHECK_NE(statefullness_, kInvalidOpKernelStatefulness);
    return statefullness_ == kStatelessOpKernel;
  }

  virtual std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const {
    return std::shared_ptr<OpKernelState>();
  }

  void Run(KernelComputeContext* ctx, OpKernelState* state) const {
    if (is_stateless()) {
      CHECK(state == nullptr);
      Compute(ctx);
    } else {
      Compute(ctx, state);
    }
  }

 protected:
  OpKernel() : statefullness_(kInvalidOpKernelStatefulness) {}

  virtual void Compute(KernelComputeContext* ctx, OpKernelState*) const {
    LOG(INFO) << "UNIMPLEMENTED";
  }
  virtual void Compute(KernelComputeContext*) const { LOG(INFO) << "UNIMPLEMENTED"; }

 private:
  template<typename T>
  friend OpKernel* NewOpKernel();

  OpKernelStatefulness statefullness_;
};

template<typename T>
OpKernel* NewOpKernel() {
  OpKernel* opkernel = new T();
  if (typeid(&OpKernel::CreateOpKernelState) == typeid(&T::CreateOpKernelState)) {
    opkernel->statefullness_ = kStatelessOpKernel;
  } else {
    opkernel->statefullness_ = kStatefulOpKernel;
  }
  return opkernel;
}

}  // namespace user_op

}  // namespace oneflow

#endif
