#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_COMPILER_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_COMPILER_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/xla/of2xla/xla_graph.h"
#include "oneflow/xla/of2xla/xla_op_context.h"
#include "oneflow/xla/of2xla/xla_utility.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/service.h"

namespace oneflow {
namespace mola {

struct CompilationResult {
  std::vector<xla::Shape> xla_input_shapes;
  // The output shape is always a tuple
  xla::Shape xla_output_shape;

  xla::XlaComputation computation;

  std::unique_ptr<xla::LocalExecutable> executable;
};

class XlaGraphCompiler {
 public:
  XlaGraphCompiler(xla::LocalClient *client, xla::XlaBuilder *builder);

  CompilationResult Compile(
      const XlaGraph *graph, const std::vector<Argument> &entry_arguments,
      const std::vector<Argument> &return_arguments,
      const std::vector<std::string> &entry_names,
      const std::vector<std::string> &return_names,
      const std::vector<xla::XlaBuilder::InputOutputAlias> &aliases);

 private:
  void BuildComputation(
      const XlaGraph *graph,
      const std::unordered_map<Argument, XlaOprand> &entry_oprands,
      const std::vector<Argument> &return_arguments, xla::Shape *output_shape,
      xla::XlaComputation *computation);

  void BuildExecutable(const CompilationResult &result,
                       std::unique_ptr<xla::LocalExecutable> *executable);

  void BuildEntryParameters(
      const std::vector<std::string> &entry_names,
      std::unordered_map<Argument, XlaOprand> *entry_oprands,
      std::vector<xla::Shape> *input_shapes);

  void SetupNodeArguments(
      const XlaNode *node,
      const std::unordered_map<std::string, Argument> &arguments,
      XlaOpContext::Param *param);

  void BuildCompilationArguments(const XlaGraph *graph,
                                 const std::vector<Argument> &entry_arguments,
                                 const std::vector<Argument> &return_arguments,
                                 const std::vector<std::string> &entry_names,
                                 const std::vector<std::string> &return_names);

 private:
  xla::LocalClient *client_;

  xla::XlaBuilder *builder_;

  std::unordered_map<std::string, Argument> arguments_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_COMPILER_H_
