#ifndef ONEFLOW_XRT_XLA_XLA_GRAPH_COMPILER_H_
#define ONEFLOW_XRT_XLA_XLA_GRAPH_COMPILER_H_

#include "oneflow/xrt/graph_compiler.h"
#include "oneflow/xrt/xla/xla_executable.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/service.h"

namespace oneflow {
namespace xrt {
namespace mola {

/*
class XlaGraphCompiler : public GraphCompiler::Impl {
 public:
  XlaGraphCompiler() = default;
  virtual ~XlaGraphCompiler() = default;

  std::shared_ptr<Executable> Compile(
      const XrtGraph *graph, const std::vector<Parameter> &entry_params,
      const std::vector<Parameter> &return_params,
      const std::vector<InputOutputAlias> &aliases) override;

 private:
  xla::LocalClient *client_;
  xla::XlaBuilder *builder_;
  util::Map<Argument, XlaOprand> entry_oprands_;
};
*/

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_XLA_GRAPH_COMPILER_H_
