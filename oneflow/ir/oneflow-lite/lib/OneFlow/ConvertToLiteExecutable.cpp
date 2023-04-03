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
#include "OneFlow/ConvertToLiteExecutable.h"

#include "OneFlow/OneFlowDialect.h"

// undefine fallthrough to fix the conflicit of flatcc and fmt
#if defined(fallthrough)
#undef fallthrough
#endif
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"
#include "OneFlow/Passes.h"
#include "OneFlow/OneFlowUtils.h"
#include "OneFlow/OneFlowLiteUtils.h"
#include "OneFlow/Transform/FoldVariable.h"
#include "OneFlow/Transform/InferPlacement.h"
#include "OneFlow/Transform/InsertTransferOp.h"
#include "OneFlow/Transform/LoweringLaunchJob.h"
#include "OneFlow/Transform/MemoryPlanning.h"
#include "OneFlow/Transform/PartitionLaunchJob.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include "schemas/executable_generated.h"
#pragma GCC diagnostic pop

namespace mlir {
namespace oneflow {

namespace lite {

static flatbuffers_vec_ref_t createLiteOpAttrs(FlatbufferBuilder& builder, Operation* op) {
  assert((llvm::dyn_cast<oneflow::UserOp>(op) || llvm::dyn_cast<UserOpCompatible>(op))
         && "the argument op is not a valid user op");
  llvm::SmallVector<oneflow_lite_AttrDef_ref_t, 4> attrDefs;
  for (auto kv : op->getAttrDictionary()) {
    auto attrName = kv.getName();
    Optional<::oneflow::AttrType> attrType =
        getUserOpAttrType(GetOpTypeName(op), attrName.strref());
    if (!attrType) { continue; }

    auto attrValue = kv.getValue();
    StringRef strAttrType;
    FlatbufferBuilder attrBuilder;
    if (attrType.value() == ::oneflow::kAtInt32) {
      strAttrType = "i32";
      serializeI32Attr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtInt64) {
      strAttrType = "i64";
      serializeI64Attr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtBool) {
      strAttrType = "bool";
      serializeBoolAttr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtFloat) {
      strAttrType = "f32";
      serializeF32Attr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtDouble) {
      strAttrType = "f64";
      serializeF64Attr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtString) {
      strAttrType = "str";
      serializeStringAttr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtShape) {
      strAttrType = "shape";
      serializeShapeAttr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtStride) {
      strAttrType = "stride";
      serializeStrideAttr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtDataType) {
      strAttrType = "dtype";
      serializeDataTypeAttr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtListInt32) {
      strAttrType = "i32s";
      serializeI32sAttr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtListInt64) {
      strAttrType = "i64s";
      serializeI64sAttr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtListFloat) {
      strAttrType = "f32s";
      serializeF32sAttr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtListDataType) {
      strAttrType = "dtypes";
      serializeDataTypesAttr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtListShape) {
      strAttrType = "shapes";
      serializeShapesAttr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtListStride) {
      strAttrType = "strides";
      serializeStridesAttr(attrBuilder, attrValue);
    } else if (attrType.value() == ::oneflow::kAtListString) {
      strAttrType = "strs";
      serializeStringsAttr(attrBuilder, attrValue);
    } else {
      llvm::errs() << "error attribute type: " << attrType.value() << "\n";
      exit(1);
    }
    oneflow_lite_AttrDef_start(builder);
    oneflow_lite_AttrDef_type_add(builder, builder.createString(strAttrType));
    oneflow_lite_AttrDef_key_add(builder, builder.createString(attrName.strref()));
    oneflow_lite_AttrDef_value_add(builder, builder.streamUint8Vec([&](llvm::raw_ostream& stream) {
      if (failed(attrBuilder.copyToStream(stream))) { return false; }
      return true;
    }));
    attrDefs.push_back(oneflow_lite_AttrDef_end(builder));
  }
  return builder.createOffsetVecDestructive(attrDefs);
}

static flatbuffers_vec_ref_t createLiteVariableOpAttrs(FlatbufferBuilder& builder, VariableOp op,
                                                       StringRef checkpointDir) {
  llvm::SmallVector<oneflow_lite_AttrDef_ref_t, 4> attrDefs;
  {
    oneflow_lite_AttrDef_start(builder);
    oneflow_lite_AttrDef_type_add(builder, builder.createString("dtype"));
    oneflow_lite_AttrDef_key_add(builder, builder.createString("dtype"));
    FlatbufferBuilder attrBuilder;
    serializeDataTypeAttr(attrBuilder, op.data_typeAttr());
    oneflow_lite_AttrDef_value_add(builder, builder.streamUint8Vec([&](llvm::raw_ostream& stream) {
      if (failed(attrBuilder.copyToStream(stream))) { return false; }
      return true;
    }));
    attrDefs.push_back(oneflow_lite_AttrDef_end(builder));
  }
  {
    oneflow_lite_AttrDef_start(builder);
    oneflow_lite_AttrDef_type_add(builder, builder.createString("shape"));
    oneflow_lite_AttrDef_key_add(builder, builder.createString("shape"));
    FlatbufferBuilder attrBuilder;
    serializeShapeAttr(attrBuilder, op.shapeAttr());
    oneflow_lite_AttrDef_value_add(builder, builder.streamUint8Vec([&](llvm::raw_ostream& stream) {
      if (failed(attrBuilder.copyToStream(stream))) { return false; }
      return true;
    }));
    attrDefs.push_back(oneflow_lite_AttrDef_end(builder));
  }
  // serialize weight data
  oneflow_lite_AttrDef_start(builder);
  oneflow_lite_AttrDef_type_add(builder, builder.createString("u8"));
  oneflow_lite_AttrDef_key_add(builder, builder.createString("value"));

  llvm::SmallString<128> inputFilename;
  llvm::sys::path::native(checkpointDir + "/" + op.op_name() + "/out", inputFilename);
  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }
  oneflow_lite_AttrDef_value_add(builder, builder.streamUint8Vec([&](llvm::raw_ostream& stream) {
    stream << input->getBuffer();
    stream.flush();
    return true;
  }));
  attrDefs.push_back(oneflow_lite_AttrDef_end(builder));
  return builder.createOffsetVecDestructive(attrDefs);
}

static oneflow_lite_OpDef_ref_t createLiteVariableOpDef(
    FlatbufferBuilder& builder, VariableOp op, llvm::DenseMap<Value, int>& valueOrdering,
    const llvm::DenseMap<StringRef, int>& deviceOrdering, StringRef checkpointDir) {
  oneflow_lite_OpDef_start(builder);
  oneflow_lite_OpDef_name_add(builder, builder.createString("constant"));
  oneflow_lite_OpDef_inputs_add(builder, 0);

  auto index = valueOrdering.try_emplace(op.output(), valueOrdering.size()).first->second;
  oneflow_lite_OpDef_outputs_add(builder,
                                 builder.createInt32Vec(llvm::SmallVector<int32_t, 4>{index}));

  oneflow_lite_OpDef_attrs_add(builder, createLiteVariableOpAttrs(builder, op, checkpointDir));

  auto it = deviceOrdering.find(op.device_tag());
  assert(it != deviceOrdering.end());
  oneflow_lite_OpDef_device_add(builder, it->second);
  return oneflow_lite_OpDef_end(builder);
}

static oneflow_lite_OpDef_ref_t createLiteOpDef(
    FlatbufferBuilder& builder, Operation* op, llvm::DenseMap<Value, int>& valueOrdering,
    const llvm::DenseMap<StringRef, int>& deviceOrdering) {
  llvm::SmallVector<size_t, 4> inputOrdering;
  for (const auto& operand : op->getOperands()) {
    auto it = valueOrdering.find(operand);
    if (it == valueOrdering.end()) {
      it = valueOrdering.try_emplace(operand, valueOrdering.size()).first;
    }
    inputOrdering.push_back(it->second);
  }
  llvm::SmallVector<size_t, 4> outputOrdering;
  for (const auto& result : op->getResults()) {
    auto it = valueOrdering.find(result);
    if (it == valueOrdering.end()) {
      it = valueOrdering.try_emplace(result, valueOrdering.size()).first;
    }
    outputOrdering.push_back(it->second);
  }
  oneflow_lite_OpDef_start(builder);
  oneflow_lite_OpDef_name_add(builder, builder.createString(GetOpTypeName(op)));
  oneflow_lite_OpDef_inputs_add(builder, builder.createInt32Vec(inputOrdering));
  oneflow_lite_OpDef_outputs_add(builder, builder.createInt32Vec(outputOrdering));

  oneflow_lite_OpDef_attrs_add(builder, createLiteOpAttrs(builder, op));

  auto device =
      op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr());
  auto it = deviceOrdering.find(device.getValue());
  assert(it != deviceOrdering.end());
  oneflow_lite_OpDef_device_add(builder, it->second);
  return oneflow_lite_OpDef_end(builder);
}

static oneflow_lite_TensorDef_ref_t createLiteTensorDef(FlatbufferBuilder& builder, Value value,
                                                        int segmentId, size_t segmentOffset) {
  TensorType type = value.getType().cast<TensorType>();
  oneflow_lite_TensorDef_start(builder);
  auto elemType = getLiteStringElementType(type.getElementType());
  if (!elemType) {
    llvm::errs() << "error tensor element type: " << type.getElementType() << "\n";
    exit(1);
  }
  oneflow_lite_TensorDef_type_add(builder, builder.createString(elemType.value()));
  oneflow_lite_TensorDef_layout_add(builder, builder.createString("default"));
  oneflow_lite_TensorDef_sizes_add(builder, builder.createInt64Vec(type.getShape()));
  oneflow_lite_TensorDef_strides_add(builder,
                                     builder.createInt64Vec(llvm::SmallVector<int64_t, 4>{}));
  oneflow_lite_TensorDef_segment_id_add(builder, segmentId);
  oneflow_lite_TensorDef_segment_offset_add(builder, segmentOffset);
  return oneflow_lite_TensorDef_end(builder);
}

static oneflow_lite_BufferSegmentDef_ref_t createLiteBufferSegmentDef(
    FlatbufferBuilder& builder, const LiteBufferSegment& segment,
    const llvm::DenseMap<StringRef, int>& deviceOrdering) {
  auto it = deviceOrdering.find(segment.device);
  assert(it != deviceOrdering.end());
  oneflow_lite_BufferSegmentDef_start(builder);
  oneflow_lite_BufferSegmentDef_size_add(builder, segment.size);
  oneflow_lite_BufferSegmentDef_device_add(builder, it->second);
  oneflow_lite_BufferSegmentDef_alignment_add(builder, static_cast<int>(segment.alignment));
  return oneflow_lite_BufferSegmentDef_end(builder);
}

LogicalResult ConvertToLiteExecutable(MLIRContext* context, ModuleOp module, ConvertOptions options,
                                      llvm::SmallVector<uint8_t, 32>* executable) {
  mlir::PassManager pm(context);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLiteFoldVariablePass());
  pm.addPass(createLiteInferPlacementPass(options.target));
  pm.addPass(createLiteInsertTransferOpPass());
  pm.addPass(createLitePartitionLaunchJobPass());
  pm.addPass(createLiteLoweringLaunchJobPass(options.checkpointDir));
  pm.addPass(createCanonicalizerPass());

  LiteBufferStrategy bufferStrategy;
  pm.addPass(createLiteMemoryPlanningPass(&bufferStrategy));
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Failed to run oneflow lite compilation passes.\n";
    return failure();
  }

  // llvm::errs() << *module << "\n";

  Operation* entryJobOp = getEntryJobOp(module);
  if (!entryJobOp) {
    llvm::errs() << "Job not found in module: " << *module;
    return failure();
  }

  auto funcName = entryJobOp->getAttrOfType<StringAttr>("sym_name");
  llvm::SmallVector<StringRef, 4> devices;
  llvm::DenseMap<StringRef, int> deviceOrdering;
  for (const auto& segment : bufferStrategy.getSegments()) {
    int ordering = deviceOrdering.size();
    if (deviceOrdering.try_emplace(segment.device, ordering).second) {
      devices.push_back(segment.device);
    }
  }
  FlatbufferBuilder builder;
  oneflow_lite_ExecutableDef_start_as_root(builder);
  oneflow_lite_ExecutableDef_version_add(builder, 0);
  oneflow_lite_ExecutableDef_name_add(builder, builder.createString(funcName.getValue()));
  oneflow_lite_ExecutableDef_devices_add(builder, builder.createStringVec(devices));

  llvm::DenseMap<Value, int> valueOrdering;
  llvm::SmallVector<int, 4> inputValueOrdering, outputValueOrdering;
  llvm::SmallVector<StringRef, 4> inputValueNames, outputValueNames;
  llvm::SmallVector<oneflow_lite_OpDef_ref_t, 4> opDefs;

  entryJobOp->walk([&](Operation* op) {
    if (!op->hasTrait<OpTrait::IsOpConfCompatible>()) { return; }
    if (auto inputOp = llvm::dyn_cast<InputOp>(op)) {
      auto it = valueOrdering.try_emplace(inputOp.output(), valueOrdering.size()).first;
      inputValueOrdering.push_back(it->second);
      inputValueNames.push_back(
          op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
              .getValue());
    } else if (auto outputOp = llvm::dyn_cast<OutputOp>(op)) {
      auto it = valueOrdering.try_emplace(outputOp.input(), valueOrdering.size()).first;
      outputValueOrdering.push_back(it->second);
      outputValueNames.push_back(
          op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
              .getValue());
    } else if (auto variableOp = llvm::dyn_cast<VariableOp>(op)) {
      opDefs.push_back(createLiteVariableOpDef(builder, variableOp, valueOrdering, deviceOrdering,
                                               options.checkpointDir));
    } else {
      opDefs.push_back(createLiteOpDef(builder, op, valueOrdering, deviceOrdering));
    }
  });
  oneflow_lite_ExecutableDef_ops_add(builder, builder.createOffsetVecDestructive(opDefs));

  llvm::SmallVector<Value, 4> orderedValues(valueOrdering.size());
  for (auto it : valueOrdering) { orderedValues[it.second] = it.first; }
  llvm::SmallVector<oneflow_lite_TensorDef_ref_t, 4> tensorDefs;
  for (auto value : orderedValues) {
    int segmentId = bufferStrategy.getValueSegmentId(value);
    size_t segmentOffset = bufferStrategy.getValueSegmentOffset(value);
    tensorDefs.push_back(createLiteTensorDef(builder, value, segmentId, segmentOffset));
  }
  oneflow_lite_ExecutableDef_operands_add(builder, builder.createOffsetVecDestructive(tensorDefs));

  oneflow_lite_ExecutableDef_inputs_add(builder, builder.createInt32Vec(inputValueOrdering));
  oneflow_lite_ExecutableDef_outputs_add(builder, builder.createInt32Vec(outputValueOrdering));
  oneflow_lite_ExecutableDef_input_names_add(builder, builder.createStringVec(inputValueNames));
  oneflow_lite_ExecutableDef_output_names_add(builder, builder.createStringVec(outputValueNames));

  llvm::SmallVector<oneflow_lite_BufferSegmentDef_ref_t, 4> segmentDefs;
  for (const auto& segment : bufferStrategy.getSegments()) {
    segmentDefs.push_back(createLiteBufferSegmentDef(builder, segment, deviceOrdering));
  }
  oneflow_lite_ExecutableDef_segments_add(builder, builder.createOffsetVecDestructive(segmentDefs));

  oneflow_lite_ExecutableDef_end_as_root(builder);

  size_t packedSize = flatcc_builder_get_buffer_size(builder);
  executable->resize(packedSize);
  flatcc_builder_copy_buffer(builder, executable->data(), packedSize);
  return success();
}

}  // namespace lite

}  // namespace oneflow
}  // namespace mlir
