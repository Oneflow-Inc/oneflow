#ifndef ONEFLOW_ENGINE_XLA_OF2XLA_XLA_OP_COMPILER_H_
#define ONEFLOW_ENGINE_XLA_OF2XLA_XLA_OP_COMPILER_H_

#include "oneflow/engine/xla/of2xla/xla_op_context.h"

namespace oneflow {
namespace mla {

class XlaOpCompiler {
 public:
  XlaOpCompiler() = default;

  virtual void Compile(XlaOpContext *ctx) = 0;

 private:
  // Something
};

}  // namespace mla
}  // namespace oneflow

#endif  // ONEFLOW_ENGINE_XLA_OF2XLA_XLA_OP_COMPILER_H_
