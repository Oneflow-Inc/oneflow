#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_compiler.h"
#include "oneflow/core/compiler/of2xla/xla_launch_kernel.h"

namespace oneflow {

static ParallelContext LocalParallelContext() {
  ParallelContext parallel_ctx;
  parallel_ctx.set_parallel_id(0);
  parallel_ctx.set_parallel_num(1);
  parallel_ctx.set_policy(kDataParallel);
  return parallel_ctx;
}

template <DeviceType device_type, typename T>
void XlaLaunchKernel<device_type, T>::ForwardDataContent(
                const KernelCtx &ctx,
                std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // Prepare setup blob descs
  std::unordered_map<std::string, BlobDesc> setup_blob_descs;
  for (const auto& input_bn : this->op_attribute().input_bns()) {
    const Blob* in_blob = BnInOp2Blob(input_bn);
    const RtBlobDesc &rt_desc = in_blob->blob_desc();
    BlobDesc blob_desc(rt_desc.shape(),
                       rt_desc.data_type(),
                       rt_desc.has_data_id_field(),
                       rt_desc.has_col_num_field(),
                       rt_desc.max_col_num());

    const LogicalBlobId& lbi = this->BnInOp2Lbi(input_bn);
    std::string blob_name = BlobName(lbi);
    setup_blob_descs.emplace(blob_name, blob_desc);
  }

  bool force_compile = true;
  ParallelContext parallel_ctx = LocalParallelContext();
  mola::XlaCompiler compiler(this->op_conf(), device_type, parallel_ctx,
                             setup_blob_descs, force_compile);
  compiler.Compile();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kXlaLaunchConf, XlaLaunchKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
