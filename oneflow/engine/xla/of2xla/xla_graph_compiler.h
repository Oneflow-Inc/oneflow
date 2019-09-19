#ifndef ONEFLOW_ENGINE_XLA_OF2XLA_XLA_GRAPH_COMPILER_H_
#define ONEFLOW_ENGINE_XLA_OF2XLA_XLA_GRAPH_COMPILER_H_

#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/engine/xla/of2xla/xla_utility.h"
#include "oneflow/engine/xla/of2xla/xla_op_context.h"
#include "oneflow/engine/xla/of2xla/xla_graph.h"

namespace oneflow {
namespace mola {

struct CompilationResult {
  std::vector<xla::Shape> xla_input_shapes;
  // The output shape is always a tuple
  xla::Shape xla_output_shape;

  xla::XlaComputation computation;

  bool alias_input_output;

  std::unique_ptr<xla::LocalExecutable> executable;
};

class XlaGraphCompiler {
 public:
  XlaGraphCompiler(xla::LocalClient *client, xla::XlaBuilder *builder,
                   XlaGraph *graph, ParallelContext parallel_ctx,
                   const std::vector<Blob *> &entry_blobs,
                   const std::vector<std::string> &entry_blob_names,
                   const std::vector<std::string> &return_blob_names,
                   const bool alias_input_output);

  CompilationResult Compile();

  void BuildComputation(
      const std::unordered_map<Argument, XlaOprand> &entry_oprands,
      const std::vector<Argument> &return_arguments,
      xla::Shape *output_shape, xla::XlaComputation *computation);

  void BuildExecutable(const CompilationResult &result,
                       std::unique_ptr<xla::LocalExecutable> *executable);

 private:
  void SetupEntryOprands(
      const std::vector<std::string> &entry_names,
      std::unordered_map<Argument, XlaOprand> *entry_oprands,
      std::vector<xla::Shape> *input_shapes);

  void SetupParamArguments(
      const XlaNode *node,
      const std::unordered_map<std::string, Argument> &arguments,
      XlaOpContext::Param *param);

 private:
  xla::LocalClient *client_;

  xla::XlaBuilder *builder_;

  XlaGraph *graph_;

  std::vector<std::string> entry_names_;
  std::vector<std::string> return_names_;

  bool alias_input_output_;

  std::unordered_map<std::string, Argument> arguments_;
};

}  // namespace mola
}  // namespace oneflow

#endif  //ONEFLOW_ENGINE_XLA_OF2XLA_XLA_GRAPH_COMPILER_H_  
