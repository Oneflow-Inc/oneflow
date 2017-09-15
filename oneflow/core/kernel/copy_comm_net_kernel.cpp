#include "oneflow/core/kernel/copy_comm_net_kernel.h"
#include "oneflow/core/comm_network/data_comm_network.h"

namespace oneflow {

void CopyCommNetKernel::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)>) const {
  auto other_val = static_cast<std::tuple<void*, const void*, const void*>*>(
      kernel_ctx.other);
  void* stream_id = std::get<0>(*other_val);
  const void* readable_token = std::get<1>(*other_val);
  const void* writeable_token = std::get<2>(*other_val);
  DataCommNet::Singleton()->Read(stream_id, readable_token, writeable_token);
}

COMMAND(AddKernelCreator(OperatorConf::kCopyCommNetConf,
                         []() { return new CopyCommNetKernel; }));

}  // namespace oneflow
