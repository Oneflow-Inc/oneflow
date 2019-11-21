#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

namespace user_op {

KernelContext::KernelContext(const KernelCtx& ctx, Blob4ArgNameAndIndexFn fn)
    : device_ctx_(ctx.device_ctx), fn_(fn) {}

Blob* KernelContext::Blob4ArgNameAndIndex(const std::string& arg_name, int32_t index) {
  return fn_(arg_name, index);
}

}  // namespace user_op

}  // namespace oneflow
