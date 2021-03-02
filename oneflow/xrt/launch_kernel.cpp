/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/xrt/launch_kernel.h"
#include "oneflow/xrt/api.h"
#include "oneflow/xrt/compilation_cache.h"
#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/graph_compiler.h"
#include "oneflow/xrt/platform.h"
#include "oneflow/xrt/utility/env.h"

// General executable setup.
DEFINE_int64(max_workspace_bytes, EnvToInt64(FLAGS_max_workspace_bytes, -1),
             "Maximum temporary workspace bytes.");
// TENSORRT executable setup.
DEFINE_int32(max_batch_size, EnvToInt(FLAGS_max_batch_size, 1),
             "Maximum batch size for builder of TENSORRT engine.");

DECLARE_bool(tensorrt_fp16);
DECLARE_bool(tensorrt_int8);
DECLARE_string(int8_calibration);

namespace oneflow {
namespace xrt {
static Parameter BuildParameter(const Blob &blob, const std::string &name) {
  const auto &desc = blob.blob_desc();
  return Parameter(name, const_cast<void *>(blob.dptr<void>()), desc.body_shape(),
                   desc.data_type());
}
}  // namespace xrt

template<DeviceType device_type>
void BlobDescGetter<device_type>::DumpEntryBlobDescTo(
    std::unordered_map<std::string, BlobDesc> *entry_blob_desc) const {
  const auto &launch_conf = kernel_->op_conf().xrt_launch_conf();
  const auto &io_mapping = launch_conf.input_output_mapping();

  for (const auto &bn : kernel_->op_attribute().input_bns()) {
    const RtBlobDesc &runtime_desc = get_blob_fn_(bn)->blob_desc();
    BlobDesc blob_desc(kernel_->job_desc().DefaultDataType());
    blob_desc.mut_shape() = runtime_desc.body_shape();
    blob_desc.set_data_type(runtime_desc.data_type());
    blob_desc.set_is_dynamic(runtime_desc.is_dynamic());
    // Map blob_name to function's input name.
    std::string blob_name = xrt::BlobIdToName(kernel_->BnInOp2Lbi(bn));
    // CHECK_GT(io_mapping.count(blob_name), 0);
    const std::string &mapping_name = io_mapping.at(blob_name);
    entry_blob_desc->emplace(mapping_name, std::move(blob_desc));
  }
}

template<DeviceType device_type>
xrt::Executable *XrtLaunchKernel<device_type>::BuildExecutable(
    const std::vector<xrt::Parameter> &entry_params,
    const std::vector<xrt::Parameter> &return_params,
    const std::vector<xrt::InputOutputAlias> &aliases, const int device_ordinal) const {
  if (!compilation_cache_) { compilation_cache_.reset(new xrt::CompilationCache); }

  xrt::Executable *executable = nullptr;
  xrt::Signature signature =
      xrt::ComputeSignature(this->op_conf().name(), device_ordinal, entry_params);
  bool force_compile = false;
  if (!force_compile) { executable = compilation_cache_->GetRecord(signature); }

  if (!executable) {
    VLOG(2) << "Build executable for launch op " << this->op_conf().name();
    const auto &launch_conf = this->op_conf().xrt_launch_conf();
    auto graph = xrt::BuildXrtGraph(launch_conf.function(), device_type, this->job_desc());
    {
      // Run InferShape pass
      const auto &parallel_ctx = this->kernel_conf().xrt_launch_conf().parallel_ctx();
      const OpAttribute& op_attribute = this->kernel_conf().op_attribute();
      CHECK(op_attribute.has_parallel_conf_signature() && op_attribute.parallel_conf_signature().has_op_parallel_conf());
      const auto &parallel_desc = ParallelDesc(op_attribute.parallel_conf_signature().op_parallel_conf());
      const auto &sbp_signatures = launch_conf.sbp_signatures();
      const auto &lbn2logical_blob_desc = launch_conf.lbn2logical_blob_desc();

      std::unordered_map<std::string, BlobDesc> entry_blob_descs;
      desc_getter_.DumpEntryBlobDescTo(&entry_blob_descs);
      auto options = xrt::CreateDefaultXrtPassOptions();
      xrt::RunXrtPass("InferShape", graph.get(), options, &this->job_desc(), &parallel_ctx, &parallel_desc,
                      &sbp_signatures, &lbn2logical_blob_desc, &entry_blob_descs);
      // Update argument meta data
      // xrt::RunXrtPass("UpdateArgMetaData", graph.get(), options,
      //                 &this->job_desc());
    }
    xrt::XrtEngine engine = xrt::StringToXrtEngine(launch_conf.engine());
    xrt::XrtDevice device = xrt::DeviceTypeToXrtDevice(device_type);
    xrt::GraphCompiler compiler(this->op_conf().name(), engine, device, device_ordinal);
    auto result = compiler.Compile(graph.get(), entry_params, return_params, aliases);
    // Record new compilation result
    compilation_cache_->Record(signature, result);
    // Get compilation result from cache
    executable = compilation_cache_->GetRecord(signature);
  }

  return std::move(executable);
}

template<DeviceType device_type>
void XrtLaunchKernel<device_type>::MakeInputOutputAlias(
    const std::vector<xrt::Parameter> &entry_params, std::vector<xrt::Parameter> *return_params,
    std::vector<xrt::InputOutputAlias> *aliases) const {
  const auto &launch_conf = this->op_conf().xrt_launch_conf();
  const auto &mutability_table = launch_conf.input_mutability();

  for (int i = 0; i < entry_params.size(); ++i) {
    const std::string &entry_name = entry_params[i].name();
    if (mutability_table.count(entry_name) > 0) {
      aliases->push_back({{static_cast<int>(return_params->size())} /*output_index*/,
                          i /*param_number=*/,
                          {} /*param_index=*/});
      return_params->push_back(entry_params[i]);
    }
  }
}

template<DeviceType device_type>
void XrtLaunchKernel<device_type>::MappingParamsToFunctionNames(
    std::vector<xrt::Parameter> *entry_params, std::vector<xrt::Parameter> *return_params) const {
  const auto &launch_conf = this->op_conf().xrt_launch_conf();
  const auto &io_mapping = launch_conf.input_output_mapping();

  for (xrt::Parameter &param : *entry_params) {
    // CHECK_GT(io_mapping.count(param.name()), 0);
    param.set_name(io_mapping.at(param.name()));
  }
  for (xrt::Parameter &param : *return_params) {
    // CHECK_GT(io_mapping.count(param.name()), 0);
    param.set_name(io_mapping.at(param.name()));
  }
}

template<DeviceType device_type>
void XrtLaunchKernel<device_type>::ForwardDataContent(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  desc_getter_ = BlobDescGetter<device_type>(this, BnInOp2Blob);
  // Prepare input and output parameters
  std::vector<xrt::Parameter> entry_params, return_params;
  for (const std::string &bn : this->op_attribute().input_bns()) {
    const LogicalBlobId &lbi = this->BnInOp2Lbi(bn);
    std::string blob_name = xrt::BlobIdToName(lbi);
    xrt::Parameter input = xrt::BuildParameter(*BnInOp2Blob(bn), blob_name);
    entry_params.push_back(input);
  }
  for (const std::string &bn : this->op_attribute().output_bns()) {
    const LogicalBlobId &lbi = this->BnInOp2Lbi(bn);
    std::string blob_name = xrt::BlobIdToName(lbi);
    xrt::Parameter output = xrt::BuildParameter(*BnInOp2Blob(bn), blob_name);
    return_params.push_back(output);
  }

  xrt::XrtDevice device = xrt::DeviceTypeToXrtDevice(device_type);
  int device_ordinal = xrt::platform::GetDeviceId(device);
  std::vector<xrt::InputOutputAlias> aliases;
  MakeInputOutputAlias(entry_params, &return_params, &aliases);
  // Mapping parameter names to function input and output names.
  MappingParamsToFunctionNames(&entry_params, &return_params);
  // Build executable.
  auto executable = BuildExecutable(entry_params, return_params, aliases, device_ordinal);
  if (!executable) { LOG(FATAL) << "Executable is built failed."; }
  // Run executable.
  xrt::ExecutableRunOptions run_options;
  run_options.device_ordinal = device_ordinal;
  run_options.return_params = return_params;
  bool block_until_done = true;
  if (device_type == DeviceType::kGPU) {
    run_options.stream = ctx.device_ctx->cuda_stream();
    run_options.device_memory_limit = FLAGS_max_workspace_bytes;
    block_until_done = false;
  }
  if (executable->engine() == xrt::XrtEngine::TENSORRT) {
    CHECK_EQ(device_type, DeviceType::kGPU);
    run_options.max_batch_size = FLAGS_max_batch_size;
    run_options.tensorrt_fp16 = FLAGS_tensorrt_fp16;
    run_options.tensorrt_int8 = FLAGS_tensorrt_int8;
    run_options.tensorrt_int8_calibration = FLAGS_int8_calibration;
  }
  bool status = executable->Run(entry_params, run_options, block_until_done);
  CHECK(status) << "Executable is running failed.";

  const std::vector<xrt::Parameter> &results = executable->Results();
  CHECK_EQ(results.size(), return_params.size());
  for (int i = 0; i < results.size(); ++i) { CHECK_EQ(results[i].data(), return_params[i].data()); }
}

// ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kXrtLaunchConf, XrtLaunchKernel,
//                            FLOATING_DATA_TYPE_SEQ);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kXrtLaunchConf, XrtLaunchKernel);

}  // namespace oneflow
