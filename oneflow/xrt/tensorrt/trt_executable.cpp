#include "cuda_runtime.h"

#include "oneflow/xrt/tensorrt/trt_executable.h"

namespace oneflow {
namespace xrt {

namespace tensorrt {

bool TrtExecutable::CreateExecutableEngine(
    const ExecutableRunOptions &run_options) {
  if (!builder_ || !network_) {
    return false;
  }
  auto build_config =
      nv::unique_ptr<nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());
  // build_config->setMaxWorkspaceSize();
  // build_config->setInt8Calibrator();
  // nvinfer1::BuilderFlags flags = 0U;
  // flags |= (1U << int(nvinfer1::BuilderFlag::kFP16));
  // flags |= (1U << int(nvinfer1::BuilderFlag::kINT8));
  // flags |= (1U << int(nvinfer1::BuilderFlag::kREFIT));
  // build_config->setFlags(flags);
  // builder_->setMaxBatchSize();
  engine_.reset(builder_->buildEngineWithConfig(*network_, *build_config));
  return true;
}

bool TrtExecutable::Execute(int batch_size, void **buffers, void *stream,
                            bool block_until_done) {
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);
  execution_context_->enqueue(batch_size, buffers, cu_stream, nullptr);
  if (block_until_done) {
    CHECK_EQ(cudaSuccess, cudaStreamSynchronize(cu_stream));
  }
  return true /* Success */;
}

bool TrtExecutable::Run(const std::vector<Parameter> &inputs,
                        const ExecutableRunOptions &run_options,
                        bool block_until_done) {
  // TODO(hjchen2)
  if (!execution_context_ && !engine_) {
    CHECK(CreateExecutableEngine(run_options))
        << "Cannot create TensorRT executanble engine.";
  }
  if (!execution_context_) {
    execution_context_.reset(engine_->createExecutionContext());
  }

  int batch_size = 1;
  void **buffers = nullptr;
  return Execute(batch_size, buffers, run_options.stream, block_until_done);
}

}  // namespace tensorrt

}  // namespace xrt
}  // namespace oneflow
