#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

namespace user_op {

KernelContext::KernelContext(DeviceCtx* device_ctx, Arg2Blob&& blobs,
                             UserOpConfWrapper&& user_op_conf)
    : device_ctx_(device_ctx), blobs_(std::move(blobs)), user_op_conf_(std::move(user_op_conf)) {}

Blob* KernelContext::Blob4ArgNameAndIndex(const std::string& arg_name, int32_t index) {
  auto it = blobs_.find(std::make_pair(arg_name, index));
  if (it == blobs_.end()) { return nullptr; }
  return &(it->second);
}

}  // namespace user_op

}  // namespace oneflow
