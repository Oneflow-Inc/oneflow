#ifndef ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

class KernelCtx;

namespace user_op {

class KernelInitContext final {
 public:
  KernelInitContext() = default;
  ~KernelInitContext() = default;
  explicit KernelInitContext(const KernelInitContext&) {}

 private:
};

using Blob4ArgNameAndIndexFn = std::function<Blob*(const std::string&, int32_t)>;

class KernelContext final {
 public:
  KernelContext() = default;
  ~KernelContext() = default;
  explicit KernelContext(const KernelCtx&, Blob4ArgNameAndIndexFn fn);

  Blob* Blob4ArgNameAndIndex(const std::string& arg_name, int32_t index);
  DeviceCtx* device_ctx() const { return device_ctx_; }

 private:
  DeviceCtx* device_ctx_;
  Blob4ArgNameAndIndexFn fn_;
};

class OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernel);
  virtual ~OpKernel() = default;

  void Init(const KernelInitContext&);
  virtual void Compute(const KernelContext&) = 0;

 protected:
  OpKernel() = default;

 private:
};

}  // namespace user_op

}  // namespace oneflow

#endif
