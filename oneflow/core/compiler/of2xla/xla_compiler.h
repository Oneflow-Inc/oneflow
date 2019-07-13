#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_COMPILER_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_COMPILER_H_

#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_op_context.h"
#include "oneflow/core/compiler/of2xla/xla_graph.h"

namespace oneflow {
namespace mola {

struct CompilationResult {
  std::vector<xla::Shape> xla_input_shapes;
  // The output shape is always a tuple
  xla::Shape xla_output_shape;

  xla::XlaComputation computation;

  std::unique_ptr<xla::LocalExecutable> executable;
};

class XlaCompiler {
 public:
  XlaCompiler(xla::LocalClient *client, xla::XlaBuilder *builder,
              const XlaLaunchOpConf &launch_conf,
              DeviceType device_type, ParallelContext parallel_ctx,
              const std::vector<Blob *> &entry_blobs,
              const std::vector<std::string> &entry_blob_names,
              const std::vector<std::string> &return_blob_names,
              bool force_compile);

  CompilationResult Compile();

  void BuildComputation(
      const std::unordered_map<Argument, XlaOprand> &entry_oprands,
      const std::vector<Argument> &return_arguments,
      xla::Shape *output_shape, xla::XlaComputation *computation);

  void BuildExecutable(const CompilationResult &result,
                       std::unique_ptr<xla::LocalExecutable> *executable);

 private:
  void SetupEntryOprands(
      std::unordered_map<Argument, XlaOprand> *entry_oprands,
      std::vector<xla::Shape> *input_shapes);

  void SetupParamArguments(
      const XlaNode *node,
      const std::unordered_map<std::string, Argument> &arguments,
      XlaOpContext::Param *param);

  std::shared_ptr<XlaGraph> graph_;

  xla::LocalClient *client_;

  xla::XlaBuilder *builder_;

  std::vector<std::string> entry_names_;
  std::vector<std::string> return_names_;
  std::unordered_map<std::string, Argument> arguments_;
  bool force_compile_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_COMPILER_H_
