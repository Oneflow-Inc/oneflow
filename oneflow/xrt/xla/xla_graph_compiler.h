#ifndef ONEFLOW_XRT_XLA_XLA_GRAPH_COMPILER_H_
#define ONEFLOW_XRT_XLA_XLA_GRAPH_COMPILER_H_

#include "oneflow/xrt/graph_compiler.h"
#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/xla_executable.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/service.h"

namespace oneflow {
namespace xrt {
namespace mola {

class XlaGraphCompiler : public GraphCompiler::Impl {
 public:
  XlaGraphCompiler() = default;
  virtual ~XlaGraphCompiler() = default;

  std::shared_ptr<Executable> Compile(
      const XrtGraph *graph, const std::vector<Parameter> &entry_params,
      const std::vector<Parameter> &return_params,
      const std::vector<InputOutputAlias> &aliases) override;

 private:
  void SetupContextParam(const XrtNode *node,
                         const util::Map<Argument, Operand> &all_outputs,
                         OpContext::Param *context_param);

  void BuildComputation(const XrtGraph *graph,
                        const std::vector<Argument> &return_args,
                        xla::Shape *output_shape,
                        xla::XlaComputation *computation);

  void BuildEntryParameters(const std::vector<Parameter> &entry_params,
                            std::vector<xla::Shape> *input_shapes);

  std::shared_ptr<Executable> BuildExecutable(
      const std::vector<xla::Shape> &xla_input_shapes,
      const xla::Shape &xla_output_shape,
      const xla::XlaComputation &computation);

  Argument ArgFromParameter(const Parameter &param);

 private:
  bool use_meta_data_ = true;

  xla::LocalClient *client_;
  xla::XlaBuilder *builder_;

  util::Map<std::string, Argument> arguments_;

  util::Map<Argument, Operand> entry_operands_;
};

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_XLA_GRAPH_COMPILER_H_
