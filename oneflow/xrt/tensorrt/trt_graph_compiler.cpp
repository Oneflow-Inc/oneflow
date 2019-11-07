#include "oneflow/xrt/tensorrt/trt_graph_compiler.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

void TrtGraphCompiler::SetupKernelContextParam(
    const XrtNode *node, OpKernelContext::Param *context_param) {

}

std::shared_ptr<Executable> TrtGraphCompiler::Compile(
      const XrtGraph *graph, const std::vector<Parameter> &entry_params,
      const std::vector<Parameter> &return_params,
      const std::vector<InputOutputAlias> &aliases) {
  algorithm::TopologyVisit(*graph, [&](const XrtNode *node) {
    OpKernelContext::Param param;
    SetupKernelContextParam(node, &param);

    OpKernelContext context(param);
    // Do compile
    auto op_kernel = BuildOpKernel(node->type());
    op_kernel->Compile(&context);

    // Post process
    
  }

  nv::unique_ptr<nvinfer1::ICudaEngine> engine;
  engine.reset(builder_->buildCudaEngine(*network_));

  return std::make_shared<TrtExecutable>(engine);
}

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
