#include "cuda_runtime.h"

#include "oneflow/xrt/tensorrt/trt_executable.h"

namespace oneflow {
namespace xrt {

namespace tensorrt {

bool TrtExecutable::CreateExecutableEngine(const ExecutableRunOptions &run_options,
                                           const int batch_size) {
  if (!builder_ || !network_) { return false; }
  auto build_config = nv::unique_ptr<nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());
  int64_t max_workspace_size = 1U << 24;  // 16MiB
  if (run_options.device_memory_limit > 0) { max_workspace_size = run_options.device_memory_limit; }
  build_config->setMaxWorkspaceSize(max_workspace_size);

  nvinfer1::BuilderFlags flags = 0U;
  if (run_options.tensorrt_fp16) {
    if (builder_->platformHasFastFp16()) {
      flags |= (1U << int(nvinfer1::BuilderFlag::kFP16));
      // It does not guarantee using half precision if only set kFP16 flag,
      // but you can set kSTRICT_TYPES to force using half precision.
      // flags |= (1U << int(nvinfer1::BuilderFlag::kSTRICT_TYPES));
    } else {
      LOG(INFO) << "TensorRT couldn't use fp16 precision since the GPU "
                   "hardware does not support.";
    }
  }
  // flags |= (1U << int(nvinfer1::BuilderFlag::kINT8));
  // flags |= (1U << int(nvinfer1::BuilderFlag::kREFIT));
  build_config->setFlags(flags);
  // build_config->setInt8Calibrator();

  int32_t max_batch_size = std::max(run_options.max_batch_size, batch_size);
  builder_->setMaxBatchSize(max_batch_size);
  // builder_->setGpuAllocator();
  engine_.reset(builder_->buildEngineWithConfig(*network_, *build_config));
  return true;
}

bool TrtExecutable::ExecuteEngine(int batch_size, void **buffers, void *stream,
                                  bool block_until_done) {
  if (!execution_context_) { execution_context_.reset(engine_->createExecutionContext()); }
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);
  bool status =
      // execution_context_->enqueue(batch_size, buffers, cu_stream, nullptr);
      execution_context_->enqueueV2(buffers, cu_stream, nullptr);
  if (block_until_done) { CHECK_EQ(cudaSuccess, cudaStreamSynchronize(cu_stream)); }
  return status;
}

bool TrtExecutable::Run(const std::vector<Parameter> &inputs,
                        const ExecutableRunOptions &run_options, bool block_until_done) {
  // TODO(hjchen2)
  if (!execution_context_ && !engine_) {
    CHECK(CreateExecutableEngine(run_options)) << "Cannot create TensorRT executanble engine.";
  }
  // All return params are results of the executable.
  this->results_ = run_options.return_params;

  util::Map<std::string, const Parameter *> all_params;
  for (const Parameter &input : inputs) { all_params.emplace(input.name(), &input); }
  for (const Parameter &output : this->results_) { all_params.emplace(output.name(), &output); }

  const int num_bindings = engine_->getNbBindings();
  std::vector<const Parameter *> binding_params(num_bindings);
  std::vector<void *> buffers(num_bindings);
  for (int i = 0; i < num_bindings; ++i) {
    const char *binding_name = engine_->getBindingName(i);
    CHECK_GT(all_params.count(binding_name), 0);
    binding_params[i] = all_params.at(binding_name);
    buffers[i] = binding_params[i]->data();
  }
  // TODO(hjchen2): Check batch size is same for all binding parameters.
  const int batch_size = binding_params[0]->shape().At(0);
  if (batch_size > engine_->getMaxBatchSize()) {
    LOG(WARNING) << "Rebuild engine since the maximum batch size " << engine_->getMaxBatchSize()
                 << " is less than the input batch size " << batch_size;
    CHECK(CreateExecutableEngine(run_options, batch_size))
        << "Failed to create engine with batch size " << batch_size;
    execution_context_.reset(engine_->createExecutionContext());
  }
  return ExecuteEngine(batch_size, buffers.data(), run_options.stream, block_until_done);
}

}  // namespace tensorrt

}  // namespace xrt
}  // namespace oneflow
