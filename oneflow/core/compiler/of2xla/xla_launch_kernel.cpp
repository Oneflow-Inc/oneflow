#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_backend.h"
#include "oneflow/core/compiler/of2xla/xla_graph_compiler.h"
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
  xla::XlaBuilder builder(this->op_conf().name());
  mola::XlaLaunchGraph graph(this->op_conf().xla_launch_conf());

  for (auto *node : graph.Nodes()) {
    node->set_backend(mola::Backend<device_type>::to_string());
  }

  // Prepare setup blob descs
  std::unordered_map<std::string, BlobDesc> blob_descs;

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
    blob_descs.emplace(blob_name, blob_desc);
  }

  ParallelContext parallel_ctx = LocalParallelContext();
  graph.InferBlobDescs(&blob_descs, &parallel_ctx);

  bool force_compile = true;
  mola::CompileContext compile_ctx(&graph, &builder, blob_descs, force_compile);

  mola::XlaGraphCompiler graph_compiler;
  graph_compiler.Compile(&compile_ctx);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kXlaLaunchConf, XlaLaunchKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
