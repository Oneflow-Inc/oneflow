#ifndef ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace user_op {

class KernelInitCtx final {
 public:
  KernelInitCtx() = default;
  ~KernelInitCtx() = default;
  explicit KernelInitCtx(const KernelInitCtx&);

 private:
};

class KernelCtx final {
 public:
  KernelCtx() = default;
  ~KernelCtx() = default;
  explicit KernelCtx(const KernelCtx&);

  Blob* Blob4ArgNameAndIndex(const std::string&, int32_t);
  DeviceCtx* device_ctx() const;

 private:
  DeviceCtx* device_ctx_;
};

class OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernel);
  virtual ~OpKernel() = default;

  void Init(const KernelInitCtx&);
  virtual void Compute(const KernelCtx&) = 0;

 protected:
  OpKernel() = default;

 private:
};

}  // namespace user_op

}  // namespace oneflow

#endif
