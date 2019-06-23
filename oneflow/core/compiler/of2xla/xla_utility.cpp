#include "oneflow/core/operator/op_conf.pb.h"
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

std::string ExtractOpTypeAsString(const OperatorConf &conf) {
  const auto it = op_type2string_map.find(conf.op_type_case());
  if (it != op_type2string_map.end()) {
    return it->second;
  } else {
    // Return empty if the operator is not in the translation map
    return NoneString;
  }
}

}  // namespace mola
}  // namespace oneflow
