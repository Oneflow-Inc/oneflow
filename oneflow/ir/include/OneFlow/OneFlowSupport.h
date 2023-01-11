/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWSUPPORT_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWSUPPORT_H_

#include <string>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "OneFlow/OneFlowEnums.h.inc"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/tensor.h"
// This include is not necessary now, but it is here for testing the namespace collision
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace mlir {

namespace oneflow {

namespace support {

const ::oneflow::UserOpDef& getUserOpDef(const std::string& op_type_name);
static const std::vector<std::string>* inputKeys() {
  static std::vector<std::string> val({"in"});
  return &val;
}

std::vector<std::string> GetInputKeys(const std::string& op_type_name);

std::vector<std::string> GetOutputKeys(const std::string& op_type_name);

mlir::DenseElementsAttr TensorToDenseElementsAttr(
    const std::shared_ptr<::oneflow::one::Tensor>& tensor, MLIRContext* ctx);

std::shared_ptr<::oneflow::one::Tensor> DenseElementsAttrToTensor(
    const mlir::Attribute& attr, const mlir::Attribute& device_tag,
    const mlir::Attribute& device_name);
void DenseElementsAttrToTensor(const mlir::Attribute& attr, const mlir::Attribute& device_tag,
                               const mlir::Attribute& device_name,
                               std::shared_ptr<::oneflow::one::Tensor>& tensor);

FailureOr<::oneflow::DataType> FromMLIRTypeToOFDataType(Type mlir_type);
FailureOr<::oneflow::DataType> FromMLIRDataTypeToOFDataType(::mlir::oneflow::DataType data_type);
FailureOr<::oneflow::DataType> FromMLIRAttrToOFDataType(Attribute attr);

}  // namespace support

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWSUPPORT_H_
