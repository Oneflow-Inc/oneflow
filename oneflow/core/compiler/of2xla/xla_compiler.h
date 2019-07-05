#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_COMPILER_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_COMPILER_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_op_context.h"
#include "oneflow/core/compiler/of2xla/xla_graph.h"

namespace oneflow {
namespace mola {

class XlaCompiler {
 public:
  XlaCompiler(const OperatorConf &op_conf, DeviceType device_type,
              ParallelContext parallel_ctx,
              const std::unordered_map<std::string, BlobDesc> &setup_blob_descs,
              bool force_compile);

  void Compile();

  void BuildComputation(
      const std::unordered_map<Argument, XlaOprand> &entry_oprands);

  void BuildExecutable();

 private:
  void SetupEntryOprands(
      std::unordered_map<Argument, XlaOprand> *entry_oprands);

  void SetupParamArguments(
      const XlaNode *node,
      const std::unordered_map<std::string, Argument> &arguments,
      XlaOpContext::Param *param);

  std::shared_ptr<XlaGraph> graph_;

  std::shared_ptr<xla::XlaBuilder> builder_;
  
  std::vector<std::string> entry_names_;
  std::unordered_map<std::string, Argument> arguments_;
  bool force_compile_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_COMPILER_H_
