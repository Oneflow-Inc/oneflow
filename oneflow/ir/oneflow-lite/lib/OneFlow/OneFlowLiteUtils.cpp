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
#include "OneFlow/OneFlowLiteUtils.h"

#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"
#include "OneFlow/OneFlowUtils.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include "schemas/executable_generated.h"
#include "schemas/attributes/bool_generated.h"
#include "schemas/attributes/f32_generated.h"
#include "schemas/attributes/f32s_generated.h"
#include "schemas/attributes/f64_generated.h"
#include "schemas/attributes/i32_generated.h"
#include "schemas/attributes/i32s_generated.h"
#include "schemas/attributes/i64_generated.h"
#include "schemas/attributes/i64s_generated.h"
#include "schemas/attributes/shape_generated.h"
#include "schemas/attributes/shapes_generated.h"
#include "schemas/attributes/str_generated.h"
#include "schemas/attributes/strs_generated.h"
#pragma GCC diagnostic pop

namespace mlir {
namespace oneflow {

namespace lite {

Operation* getEntryJobOp(ModuleOp module) { return getEntryJobOp(module.getOperation()); }

Operation* getEntryJobOp(Operation* op) {
  Operation* entry = nullptr;
  op->walk([&](oneflow::Job job) -> WalkResult {
    entry = job.getOperation();
    return WalkResult::advance();
  });
  return entry;
}

StringAttr getValueDevice(Value value) {
  StringAttr device;
  Operation* op = value.getDefiningOp();
  if (auto copyOp = dyn_cast<CopyOp>(op)) {
    device = copyOp.device_typeAttr();
  } else {
    device = value.getDefiningOp()->getAttrOfType<StringAttr>(
        OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr());
  }
  return device;
}

Optional<StringRef> getLiteStringElementType(Type type) {
  assert(type.isIntOrFloat());
  if (type.isF16()) {
    return StringRef("f16");
  } else if (type.isBF16()) {
    return StringRef("bf16");
  } else if (type.isF32()) {
    return StringRef("f32");
  } else if (type.isF64()) {
    return StringRef("f64");
  } else if (type.isSignedInteger()) {
    int bitwidth = type.getIntOrFloatBitWidth();
    return StringRef("i" + llvm::Twine(bitwidth).str());
  } else if (type.isUnsignedInteger()) {
    int bitwidth = type.getIntOrFloatBitWidth();
    return StringRef("u" + llvm::Twine(bitwidth).str());
  } else {
    return llvm::None;
  }
}

Optional<StringRef> getLiteStringElementType(::mlir::oneflow::DataType type) {
  switch (type) {
    case ::mlir::oneflow::DataType::DT_Bool: return StringRef("bool");
    case ::mlir::oneflow::DataType::DT_Char: return StringRef("char");
    case ::mlir::oneflow::DataType::DT_Float16: return StringRef("f16");
    case ::mlir::oneflow::DataType::DT_Float: return StringRef("f32");
    case ::mlir::oneflow::DataType::DT_Double: return StringRef("f64");
    case ::mlir::oneflow::DataType::DT_Int8: return StringRef("i8");
    case ::mlir::oneflow::DataType::DT_Int32: return StringRef("i32");
    case ::mlir::oneflow::DataType::DT_Int64: return StringRef("i64");
    case ::mlir::oneflow::DataType::DT_UInt8: return StringRef("u8");
    default: {
      return llvm::None;
    }
  }
}

Optional<::oneflow::AttrType> getUserOpAttrType(StringRef opName, StringRef attrName) {
  const ::oneflow::user_op::OpRegistryResult* val =
      ::oneflow::user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(opName.str());
  if (!val) {
    llvm::errs() << "unregistered user op: " << opName << "\n";
    exit(1);
  }
  ::oneflow::user_op::UserOpDefWrapper op_def(val->op_def);
  if (!op_def.IsAttrName(attrName.str())) { return llvm::None; }
  return op_def.GetAttrType(attrName.str());
}

void serializeI32Attr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_I32Def_start_as_root(builder);
  oneflow_lite_I32Def_value_add(builder, attribute.dyn_cast<IntegerAttr>().getSInt());
  oneflow_lite_I32Def_end_as_root(builder);
}

void serializeI64Attr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_I64Def_start_as_root(builder);
  oneflow_lite_I64Def_value_add(builder, attribute.dyn_cast<IntegerAttr>().getSInt());
  oneflow_lite_I64Def_end_as_root(builder);
}

void serializeBoolAttr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_BoolDef_start_as_root(builder);
  oneflow_lite_BoolDef_value_add(builder, attribute.dyn_cast<BoolAttr>().getValue());
  oneflow_lite_BoolDef_end_as_root(builder);
}

void serializeF32Attr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_F32Def_start_as_root(builder);
  oneflow_lite_F32Def_value_add(builder,
                                attribute.dyn_cast<FloatAttr>().getValue().convertToFloat());
  oneflow_lite_F32Def_end_as_root(builder);
}

void serializeF64Attr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_F64Def_start_as_root(builder);
  oneflow_lite_F64Def_value_add(builder,
                                attribute.dyn_cast<FloatAttr>().getValue().convertToDouble());
  oneflow_lite_F64Def_end_as_root(builder);
}

void serializeStringAttr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_StringDef_start_as_root(builder);
  oneflow_lite_StringDef_value_add(
      builder, builder.createString(attribute.dyn_cast<StringAttr>().getValue()));
  oneflow_lite_StringDef_end_as_root(builder);
}

void serializeShapeAttr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_ShapeDef_start_as_root(builder);
  SmallVector<int64_t, 4> shape;
  for (auto v : attribute.dyn_cast<ArrayAttr>().getValue()) {
    shape.push_back(v.dyn_cast<IntegerAttr>().getSInt());
  }
  oneflow_lite_ShapeDef_value_add(builder, builder.createInt64Vec(shape));
  oneflow_lite_ShapeDef_end_as_root(builder);
}

void serializeStrideAttr(FlatbufferBuilder& builder, Attribute attribute) {
  serializeShapeAttr(builder, attribute);
}

void serializeDataTypeAttr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_StringDef_start_as_root(builder);
  auto dtype =
      getLiteStringElementType(attribute.dyn_cast<mlir::oneflow::DataTypeAttr>().getValue());
  if (!dtype) {
    llvm::errs() << "error data type: " << attribute << "\n";
    exit(1);
  }
  oneflow_lite_StringDef_value_add(builder, builder.createString(dtype.value()));
  oneflow_lite_StringDef_end_as_root(builder);
}

void serializeI32sAttr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_I32sDef_start_as_root(builder);
  SmallVector<int32_t, 4> vec;
  for (auto v : attribute.dyn_cast<ArrayAttr>().getValue()) {
    vec.push_back(v.dyn_cast<IntegerAttr>().getSInt());
  }
  oneflow_lite_I32sDef_value_add(builder, builder.createInt32Vec(vec));
  oneflow_lite_I32sDef_end_as_root(builder);
}

void serializeI64sAttr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_I64sDef_start_as_root(builder);
  SmallVector<int64_t, 4> vec;
  for (auto v : attribute.dyn_cast<ArrayAttr>().getValue()) {
    vec.push_back(v.dyn_cast<IntegerAttr>().getSInt());
  }
  oneflow_lite_I64sDef_value_add(builder, builder.createInt64Vec(vec));
  oneflow_lite_I64sDef_end_as_root(builder);
}

void serializeF32sAttr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_F32sDef_start_as_root(builder);
  flatbuffers_float_vec_start(builder);
  for (auto v : attribute.dyn_cast<ArrayAttr>().getValue()) {
    flatbuffers_float_vec_push_create(builder, v.dyn_cast<FloatAttr>().getValue().convertToFloat());
  }
  oneflow_lite_F32sDef_value_add(builder, flatbuffers_float_vec_end(builder));
  oneflow_lite_F32sDef_end_as_root(builder);
}

void serializeDataTypesAttr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_StringsDef_start_as_root(builder);
  llvm::SmallVector<StringRef, 4> dtypes;
  for (auto v : attribute.dyn_cast<ArrayAttr>().getValue()) {
    auto dtype = getLiteStringElementType(v.dyn_cast<mlir::oneflow::DataTypeAttr>().getValue());
    if (!dtype) {
      llvm::errs() << "error data type: " << v << "\n";
      exit(1);
    }
    dtypes.push_back(dtype.getValue());
  }
  oneflow_lite_StringsDef_value_add(builder, builder.createStringVec(dtypes));
  oneflow_lite_StringsDef_end_as_root(builder);
}

void serializeShapesAttr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_ShapesDef_start_as_root(builder);
  SmallVector<oneflow_lite_ShapeDef_ref_t, 4> shapeDefs;
  for (auto v : attribute.dyn_cast<ArrayAttr>().getValue()) {
    oneflow_lite_ShapeDef_start(builder);
    SmallVector<int64_t, 4> vec;
    for (auto p : v.dyn_cast<ArrayAttr>().getValue()) {
      vec.push_back(p.dyn_cast<IntegerAttr>().getSInt());
    }
    oneflow_lite_ShapeDef_value_add(builder, builder.createInt64Vec(vec));
    shapeDefs.push_back(oneflow_lite_ShapeDef_end(builder));
  }
  oneflow_lite_ShapesDef_value_add(builder, builder.createOffsetVecDestructive(shapeDefs));
  oneflow_lite_ShapesDef_end_as_root(builder);
}

void serializeStridesAttr(FlatbufferBuilder& builder, Attribute attribute) {
  return serializeShapesAttr(builder, attribute);
}

void serializeStringsAttr(FlatbufferBuilder& builder, Attribute attribute) {
  oneflow_lite_StringsDef_start_as_root(builder);
  SmallVector<oneflow_lite_StringDef_ref_t, 4> stringDefs;
  for (auto v : attribute.dyn_cast<ArrayAttr>().getValue()) {
    oneflow_lite_StringDef_start(builder);
    oneflow_lite_StringDef_value_add(builder,
                                     builder.createString(v.dyn_cast<StringAttr>().getValue()));
    stringDefs.push_back(oneflow_lite_StringDef_end(builder));
  }
  oneflow_lite_StringsDef_value_add(builder, builder.createOffsetVecDestructive(stringDefs));
  oneflow_lite_StringsDef_end_as_root(builder);
}

}  // namespace lite

}  // namespace oneflow
}  // namespace mlir
