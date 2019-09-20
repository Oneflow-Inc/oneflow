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

  std::unique_ptr<xla::LocalExecutable> executable;
};

class XlaGraphCompiler {
 public:
  XlaGraphCompiler(xla::LocalClient *client, xla::XlaBuilder *builder);

  CompilationResult Compile(
      const XlaGraph *graph,
      const std::vector<Blob *> &entry_blobs,
      const std::vector<Blob *> &return_blobs,
      const std::vector<std::string> &entry_blob_names,
      const std::vector<std::string> &return_blob_names,
      const std::vector<xla::XlaBuilder::InputOutputAlias> &aliases);

 private:
  void BuildComputation(
      const XlaGraph *graph,
      const std::unordered_map<Argument, XlaOprand> &entry_oprands,
      const std::vector<Argument> &return_arguments,
      xla::Shape *output_shape, xla::XlaComputation *computation);

  void BuildExecutable(const CompilationResult &result,
                       std::unique_ptr<xla::LocalExecutable> *executable);

  void SetupEntryOprands(
      const std::vector<std::string> &entry_names,
      std::unordered_map<Argument, XlaOprand> *entry_oprands,
      std::vector<xla::Shape> *input_shapes);

  void SetupNodeArguments(
      const XlaNode *node,
      const std::unordered_map<std::string, Argument> &arguments,
      XlaOpContext::Param *param);

  void BuildArguments(
    const XlaGraph *graph,
    const std::vector<Blob *> &entry_blobs,
    const std::vector<Blob *> &return_blobs,
    const std::vector<std::string> &entry_blob_names,
    const std::vector<std::string> &return_blob_names);

 private:
  xla::LocalClient *client_;

  xla::XlaBuilder *builder_;

  std::unordered_map<std::string, Argument> arguments_;
};

}  // namespace mola
}  // namespace oneflow

#endif  //ONEFLOW_ENGINE_XLA_OF2XLA_XLA_GRAPH_COMPILER_H_  
