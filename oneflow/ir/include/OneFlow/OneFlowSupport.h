#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWSUPPORT_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWSUPPORT_H_

#include <string>
#include <vector>

// We need this "pure" header file to avoid the collision of MLIR and OneFlow's includes...

namespace mlir {

namespace oneflow {

namespace support {

static const std::vector<std::string>* inputKeys() {
  static std::vector<std::string> val({"in"});
  return &val;
}

std::vector<std::string> GetInputKeys(const std::string& op_type_name);

std::vector<std::string> GetOutputKeys(const std::string& op_type_name);

}  // namespace support

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWSUPPORT_H_
