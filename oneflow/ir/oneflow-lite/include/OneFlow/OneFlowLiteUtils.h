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
#ifndef ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_ONEFLOWLITEUTILS_H_
#define ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_ONEFLOWLITEUTILS_H_

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/FlatbufferUtils.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace oneflow {

namespace lite {

Operation* getEntryJobOp(ModuleOp module);
Operation* getEntryJobOp(Operation* op);

StringAttr getValueDevice(Value value);

Optional<StringRef> getLiteStringElementType(Type type);
Optional<StringRef> getLiteStringElementType(::mlir::oneflow::DataType type);

Optional<::oneflow::AttrType> getUserOpAttrType(StringRef opName, StringRef attrName);

void serializeI32Attr(FlatbufferBuilder& builder, Attribute attribute);
void serializeI64Attr(FlatbufferBuilder& builder, Attribute attribute);
void serializeBoolAttr(FlatbufferBuilder& builder, Attribute attribute);
void serializeF32Attr(FlatbufferBuilder& builder, Attribute attribute);
void serializeF64Attr(FlatbufferBuilder& builder, Attribute attribute);
void serializeStringAttr(FlatbufferBuilder& builder, Attribute attribute);
void serializeShapeAttr(FlatbufferBuilder& builder, Attribute attribute);
void serializeStrideAttr(FlatbufferBuilder& builder, Attribute attribute);
void serializeDataTypeAttr(FlatbufferBuilder& builder, Attribute attribute);
void serializeI32sAttr(FlatbufferBuilder& builder, Attribute attribute);
void serializeI64sAttr(FlatbufferBuilder& builder, Attribute attribute);
void serializeF32sAttr(FlatbufferBuilder& builder, Attribute attribute);
void serializeDataTypesAttr(FlatbufferBuilder& builder, Attribute attribute);
void serializeShapesAttr(FlatbufferBuilder& builder, Attribute attribute);
void serializeStridesAttr(FlatbufferBuilder& builder, Attribute attribute);
void serializeStringsAttr(FlatbufferBuilder& builder, Attribute attribute);

}  // namespace lite

}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_ONEFLOWLITEUTILS_H_
