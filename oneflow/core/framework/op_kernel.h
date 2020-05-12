#ifndef ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_

#include <glog/logging.h>
#include <memory>
#include "oneflow/core/framework/util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_registration.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/placement.pb.h"

namespace oneflow {

class JobDesc;

namespace user_op {

class KernelInitContext {
 public:
  virtual ~KernelInitContext() = default;

  virtual DeviceCtx* device_ctx() = 0;

  virtual DeviceType device_type() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;
  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string&, int32_t) const = 0;
  virtual const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string&, int32_t) const = 0;

  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;

  template<typename T>
  T GetAttr(const std::string& attr_name) const {
    return user_op_conf_.attr<T>(attr_name);
  }
  const UserOpConfWrapper& user_op_conf() const { return user_op_conf_; }

 protected:
  KernelInitContext(UserOpConfWrapper&& conf) : user_op_conf_(std::move(conf)) {}
  KernelInitContext(const KernelInitContext&) = delete;

 private:
  UserOpConfWrapper user_op_conf_;
};

class KernelInferContext {
 public:
  virtual ~KernelInferContext() = default;

  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;
  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string&, int32_t) const = 0;
  virtual DeviceType device_type() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;

  virtual DeviceCtx* device_ctx() = 0;
  virtual Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t arg_index) = 0;
  virtual const ShapeView& ShapeView4ArgNameAndIndex(const std::string& arg_name,
                                                     int32_t arg_index) = 0;
  virtual MutShapeView* MutShapeView4ArgNameAndIndex(const std::string& arg_name,
                                                     int32_t arg_index) = 0;

  template<typename T>
  T GetAttr(const std::string& attr_name) const {
    return user_op_conf_.attr<T>(attr_name);
  }

  virtual InferContext* MutOpInferContext() {
    UNIMPLEMENTED();
    return nullptr;
  }
  virtual const TensorDescInferFn& GetOpInferFn() const { UNIMPLEMENTED(); }

 protected:
  KernelInferContext(UserOpConfWrapper&& conf) : user_op_conf_(conf) {}
  KernelInferContext(const KernelInferContext&) = delete;

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
  virtual const JobDesc& job_desc() const = 0;

  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;

  template<typename T>
  T GetAttr(const std::string& attr_name) const {
    return user_op_conf_.attr<T>(attr_name);
  }
  const UserOpConfWrapper& user_op_conf() const { return user_op_conf_; }

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

class OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernel);
  virtual ~OpKernel() = default;

  virtual std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const {
    return std::shared_ptr<OpKernelState>();
  }

  virtual void Compute(KernelComputeContext* ctx, OpKernelState*) const { Compute(ctx); }
  virtual void Compute(KernelComputeContext*) const { LOG(INFO) << "UNIMPLEMENTED"; }
  virtual void InferShape(KernelInferContext* ctx) const;
  virtual bool AlwaysComputeWhenAllOutputsEmpty() const = 0;

 protected:
  OpKernel() = default;
};

}  // namespace user_op

}  // namespace oneflow

#endif
