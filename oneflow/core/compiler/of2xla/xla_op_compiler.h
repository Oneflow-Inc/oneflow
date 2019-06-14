#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_COMPILER_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_COMPILER_H_

#include "oneflow/core/compiler/of2xla/xla_op_context.h"

namespace oneflow {
namespace mola {

class XlaOpCompiler {
 public:
  XlaOpCompiler() = default;

  virtual void Compile(XlaOpContext *ctx) = 0;

 private:
  // Something
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_COMPILER_H_
