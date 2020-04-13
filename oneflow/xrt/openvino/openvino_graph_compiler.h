#ifndef ONEFLOW_XRT_OPENVINO_OPENVINO_GRAPH_COMPILER_H_
#define ONEFLOW_XRT_OPENVINO_OPENVINO_GRAPH_COMPILER_H_

#include "oneflow/xrt/graph_compiler.h"
#include "oneflow/xrt/openvino/ops/op_context.h"
#include "oneflow/xrt/openvino/ngraph_shape.h"
#include "oneflow/xrt/openvino/openvino_executable.h"

namespace oneflow {
namespace xrt {
namespace openvino {

class OpenvinoGraphCompiler : public GraphCompiler::Impl {
 public:
  explicit OpenvinoGraphCompiler(const std::string &name) : GraphCompiler::Impl(name) {}

  virtual ~OpenvinoGraphCompiler() = default;

  std::shared_ptr<Executable> Compile(const XrtGraph *graph,
                                      const std::vector<Parameter> &entry_params,
                                      const std::vector<Parameter> &return_params,
                                      const std::vector<InputOutputAlias> &aliases) override;

 private:
  void SetupKernelContextParam(const XrtNode *node, OpenvinoOpContext::Param *context_param);

  void PopulateEntryParams(const std::vector<Parameter> &entry_params, ngraph::ParameterVector *,
                           util::Map<std::string, int> &);

  Argument ArgFromParameter(const Parameter &param);

 private:
  util::Map<std::string, Argument> arguments_;
  util::Map<Argument, std::shared_ptr<ngraph::Node>> operands_;
};

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_OPENVINO_OPENVINO_GRAPH_COMPILER_H_
