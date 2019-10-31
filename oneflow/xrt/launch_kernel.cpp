#include "oneflow/xrt/launch_kernel.h"
#include "oneflow/xrt/api.h"
#include "oneflow/xrt/compilation_cache.h"
#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/graph_compiler.h"

namespace oneflow {

template <DeviceType device_type>
void BlobDescGetter<device_type>::DumpEntryTo(
    std::unordered_map<std::string, BlobDesc> *entry_blob_desc) {
  for (const auto &bn : kernel_->op_attribute().input_bns()) {
    const RtBlobDesc &runtime_desc = get_blob_fn_(bn)->blob_desc();
    BlobDesc blob_desc(runtime_desc.shape(), runtime_desc.data_type(),
                       runtime_desc.has_data_id_field(),
                       runtime_desc.has_col_num_field(),
                       runtime_desc.max_col_num());
    std::string blob_name = xrt::BlobIdToName(kernel_->BnInOp2Lbi(bn));
    entry_blob_desc->emplace(blob_name, std::move(blob_desc));
  }
}

template <DeviceType device_type>
xrt::Executable *XrtLaunchKernel<device_type>::BuildExecutable(
    const std::vector<xrt::Parameter> &entry_params,
    const std::vector<xrt::Parameter> &return_params,
    const std::vector<xrt::InputOutputAlias> &aliases) const {
  if (!compilation_cache_) {
    compilation_cache_.reset(new xrt::CompilationCache);
  }

  xrt::Executable *executable = nullptr;
  // TODO(hjchen2)
  // const int device_ordinal = launch_ctx->device_ordinal();
  int device_ordinal = 0;
  xrt::Signature signature = xrt::ComputeSignature(
      this->op_conf().name(), device_ordinal, entry_params);
  bool force_compile = false;
  if (!force_compile) {
    executable = compilation_cache_->GetRecord(signature);
  }

  if (!executable) {
    auto graph = xrt::BuildXrtGraph(this->op_conf().xrt_launch_conf(),
                                    device_type, this->job_desc());
    {
      // Run InferShape pass
      std::unordered_map<std::string, BlobDesc> entry_blob_descs;
      desc_getter_.DumpEntryTo(&entry_blob_descs);
      auto options = xrt::CreateDefaultXrtPassOptions();
      xrt::RunXrtPass("InferShape", graph.get(), options, &this->job_desc(),
                      &entry_blob_descs);
      // Update argument meta data
      xrt::RunXrtPass("UpdateArgMetaData", graph.get(), options,
                      &this->job_desc());
    }

    xrt::XrtEngine engine = xrt::XLA;
    xrt::XrtDevice device = xrt::DeviceTypeToXrtDevice(device_type);
    xrt::GraphCompiler compiler(this->op_conf().name(), engine, device,
                                device_ordinal);
    auto result =
        compiler.Compile(graph.get(), entry_params, return_params, aliases);
    // Record new compilation result
    compilation_cache_->Record(signature, result);
    // Get compilation result from cache
    executable = compilation_cache_->GetRecord(signature);
  }

  return std::move(executable);
}

template <DeviceType device_type>
void XrtLaunchKernel<device_type>::MakeInputOutputAlias(
    const std::vector<xrt::Parameter> &entry_params,
    std::vector<xrt::Parameter> *return_params,
    std::vector<xrt::InputOutputAlias> *aliases) const {
  const auto &launch_conf = this->op_conf().xrt_launch_conf();
  for (int i = 0; i < entry_params.size(); ++i) {
    const std::string &entry_name = entry_params[i].name();
    if (xrt::LookupMutability(launch_conf, entry_name)) {
      aliases->push_back(
          {{static_cast<int>(return_params->size())} /*output_index*/,
           i /*param_number=*/,
           {} /*param_index=*/});
      return_params->push_back(entry_params[i]);
    }
  }
}

template <DeviceType device_type>
void XrtLaunchKernel<device_type>::ForwardDataContent(
    const KernelCtx &ctx,
    std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  desc_getter_ = BlobDescGetter<device_type>(this, BnInOp2Blob);
  // Prepare input and output parameters
  std::vector<xrt::Parameter> entry_params, return_params;
  for (const auto &bn : this->op_attribute().input_bns()) {
    std::string blob_name = xrt::BlobIdToName(this->BnInOp2Lbi(bn));
    xrt::Parameter input = xrt::BuildParameter(*BnInOp2Blob(bn), blob_name);
    entry_params.push_back(input);
  }
  for (const auto &bn : this->op_attribute().output_bns()) {
    xrt::Parameter output = xrt::BuildParameter(*BnInOp2Blob(bn), bn);
    return_params.push_back(output);
  }

  std::vector<xrt::InputOutputAlias> aliases;
  MakeInputOutputAlias(entry_params, &return_params, &aliases);
  auto executable = BuildExecutable(entry_params, return_params, aliases);
  if (!executable) {
    LOG(FATAL) << "Executable is built failed.";
  }

  xrt::ExecutableRunOptions run_options;
  run_options.return_params = return_params;
  bool block_until_done = true;
  if (device_type == DeviceType::kGPU) {
    block_until_done = false;
    run_options.stream = ctx.device_ctx->cuda_stream();
  }
  bool status = executable->Run(entry_params, run_options, block_until_done);
  CHECK(status) << "Executable is running failed.";

  const std::vector<xrt::Parameter> &results = executable->Results();
  CHECK_EQ(results.size(), return_params.size());
  for (int i = 0; i < results.size(); ++i) {
    CHECK_EQ(results[i].data(), return_params[i].data());
  }
}

// ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kXrtLaunchConf, XrtLaunchKernel,
//                            FLOATING_DATA_TYPE_SEQ);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kXrtLaunchConf, XrtLaunchKernel);

}  // namespace oneflow
