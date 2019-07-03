#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_COMPILER_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_COMPILER_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_op_context.h"
#include "oneflow/core/compiler/of2xla/xla_graph.h"

namespace oneflow {
namespace mola {

class CompileContext {
 public:
  typedef std::unordered_map<std::string, BlobDesc> BlobDescMap;

  CompileContext(const XlaGraph *graph, xla::XlaBuilder *builder,
                 const BlobDescMap &infered_blob_descs, bool force_compile)
      : graph_(graph), builder_(builder), force_compile_(force_compile) {
    for (const auto &pair : infered_blob_descs) {
      LogicalBlobId lbi = BlobId(pair.first);
      arguments_.emplace(lbi, Argument(lbi, pair.second));
    }
  }
   
 private:
  friend class XlaGraphCompiler;

  const XlaGraph *graph_;

  xla::XlaBuilder *builder_;

  bool force_compile_;
  std::unordered_map<LogicalBlobId, Argument> arguments_;
};

class XlaGraphCompiler {
 public:
  XlaGraphCompiler() = default;

  void Compile(CompileContext *ctx);

  void SetupParamArguments(
                const XlaNode *node,
                const std::unordered_map<LogicalBlobId, Argument> &arguments,
                XlaOpContext::Param *param);
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_COMPILER_H_
