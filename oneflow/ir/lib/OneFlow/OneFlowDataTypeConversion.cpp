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
#include "OneFlow/OneFlowDataTypeConversion.h"
#include "OneFlow/OneFlowTypes.h"

namespace mlir {

namespace oneflow {

Type getTypeFromOneFlowDataType(MLIRContext* context, ::oneflow::DataType dt) {
  if (dt == ::oneflow::DataType::kInvalidDataType) { return InvalidElementType::get(context); }
  if (dt == ::oneflow::DataType::kChar) { return CharElementType::get(context); }
  if (dt == ::oneflow::DataType::kFloat16) { return FloatType::getF16(context); }
  if (dt == ::oneflow::DataType::kFloat) { return FloatType::getF32(context); }
  if (dt == ::oneflow::DataType::kDouble) { return FloatType::getF64(context); }
  if (dt == ::oneflow::DataType::kInt8) {
    return IntegerType::get(context, 8, IntegerType::Signed);
  }
  if (dt == ::oneflow::DataType::kInt32) {
    return IntegerType::get(context, 32, IntegerType::Signed);
  }
  if (dt == ::oneflow::DataType::kInt64) {
    return IntegerType::get(context, 64, IntegerType::Signed);
  }
  if (dt == ::oneflow::DataType::kOFRecord) { return OFRecordElementType::get(context); }
  if (dt == ::oneflow::DataType::kTensorBuffer) { return TensorBufferElementType::get(context); }
  if (dt == ::oneflow::DataType::kBool) {
    return IntegerType::get(context, 8, IntegerType::Signed);
  }
  if (dt == ::oneflow::DataType::kUInt8) {
    return IntegerType::get(context, 8, IntegerType::Unsigned);
  }
  if (dt == ::oneflow::DataType::kUInt16) {
    return IntegerType::get(context, 16, IntegerType::Unsigned);
  }
  if (dt == ::oneflow::DataType::kUInt32) {
    return IntegerType::get(context, 32, IntegerType::Unsigned);
  }
  if (dt == ::oneflow::DataType::kUInt64) {
    return IntegerType::get(context, 64, IntegerType::Unsigned);
  }
  if (dt == ::oneflow::DataType::kUInt128) {
    return IntegerType::get(context, 128, IntegerType::Unsigned);
  }
  llvm::errs() << "unsupported oneflow data type: " << dt << "\n";
  return Type();
}

}  // namespace oneflow

}  // namespace mlir
