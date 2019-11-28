#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

namespace user_op {

KernelContext::KernelContext(DeviceCtx* device_ctx, ArgNameAndIndex2Blob&& blobs)
    : device_ctx_(device_ctx), blobs_(std::move(blobs)) {}

Blob* KernelContext::Blob4ArgNameAndIndex(const std::string& arg_name, int32_t index) {
  auto it = blobs_.find(std::make_pair(arg_name, index));
  if (it == blobs_.end()) { return nullptr; }
  return it->second.get();
}

}  // namespace user_op

}  // namespace oneflow
