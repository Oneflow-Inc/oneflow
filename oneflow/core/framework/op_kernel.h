#ifndef ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_

#include "oneflow/core/framework/util.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/framework/blob.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/kernel_context.h"

namespace oneflow {

class KernelCtx;

namespace user_op {

class KernelInitContext final {};

class OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernel);
  virtual ~OpKernel() = default;

  virtual void Compute(KernelContext*) = 0;

 protected:
  OpKernel(const KernelInitContext&) {}

 private:
};

}  // namespace user_op

}  // namespace oneflow

#endif
