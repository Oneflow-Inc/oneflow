#include <string>
#include <unordered_map>
#include "tensorflow/compiler/xla/shape.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/compiler/of2xla/xla_op_compiler_registry.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"

namespace oneflow {
namespace mola {

#define OP_TYPE_CASE(op)  OperatorConf::k##op##Conf

static std::unordered_map<int32_t, std::string> op_type2string_map = {
  {OP_TYPE_CASE(Matmul),  "MatMul"},
  {OP_TYPE_CASE(Relu), "Relu"},
  {OP_TYPE_CASE(FullyConnected), "FullyConnected"},
  // TODO(hjchen2)
};

std::string ExtractOpTypeAsString(const Operator &op) {
  const OperatorConf &conf = op.op_conf();
  const auto it = op_type2string_map.find(conf.op_type_case());
  if (it != op_type2string_map.end()) {
    return it->second;
  } else {
    // Return empty if the operator is not in the translation map
    return NoneString;
  }
}

bool IsOpTypeCompiled(const std::string &backend, const std::string &type) {
  // TODO(hjchen2) should replaced with node->compiled
  return XlaOpCompilerRegistry::IsRegistered(backend, type);
}

Shape ShapeFromXlaShape(const xla::Shape &xla_shape) {
  // TODO(hjchen2)
  Shape shape;
  return shape;
}

xla::Shape XlaShapeFromShape(const Shape &shape) {
  // TODO(hjchen2)
  xla::Shape xla_shape;
  return xla_shape;
}

}  // namespace mola
}  // namespace oneflow
