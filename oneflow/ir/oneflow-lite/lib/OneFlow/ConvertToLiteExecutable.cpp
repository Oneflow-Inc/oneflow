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

#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"
#include "OneFlow/Passes.h"
#include "OneFlow/OneFlowUtils.h"
#include "OneFlow/Transform/FoldVariable.h"
#include "OneFlow/Transform/InferPlacement.h"
#include "OneFlow/Transform/InsertTransferOp.h"
#include "OneFlow/Transform/MemoryPlanning.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include "schemas/executable_generated.h"
#pragma GCC diagnostic pop

namespace mlir {
namespace oneflow {

namespace lite {

static StringRef getLiteStringElementType(Type type) {
  assert(type.isIntOrFloat());
  if (type.isF16()) {
    return "f16";
  } else if (type.isBF16()) {
    return "bf16";
  } else if (type.isF32()) {
    return "f32";
  } else if (type.isF64()) {
    return "f64";
  } else if (type.isSignedInteger()) {
    int bitwidth = type.getIntOrFloatBitWidth();
    return "i" + llvm::Twine(bitwidth).str();
  } else if (type.isUnsignedInteger()) {
    int bitwidth = type.getIntOrFloatBitWidth();
    return "u" + llvm::Twine(bitwidth).str();
  } else {
    llvm::errs() << "error type: " << type << "\n";
    return "";
  }
}

static Optional<::oneflow::AttrType> getUserOpAttrType(StringRef opName, StringRef attrName) {
  const ::oneflow::user_op::OpRegistryResult* val =
      ::oneflow::user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(opName.str());
  if (!val) {
    llvm::errs() << "unregistered user op: " << opName.str() << "\n";
    exit(1);
  }
  ::oneflow::user_op::UserOpDefWrapper op_def(val->op_def);
  if (!op_def.IsAttrName(attrName.str())) { return llvm::None; }
  return op_def.GetAttrType(attrName.str());
}

static flatbuffers_vec_ref_t createLiteUserOpAttrs(FlatbufferBuilder& builder, Operation* op) {
  assert((llvm::dyn_cast<oneflow::UserOp>(op) || llvm::dyn_cast<UserOpCompatible>(op))
         && "the argument op is not a valid user op");
  llvm::SmallVector<oneflow_lite_AttrDef_ref_t, 4> attrDefs;
  for (auto kv : op->getAttrDictionary()) {
    auto attrName = kv.getName();
    Optional<::oneflow::AttrType> attrType =
        getUserOpAttrType(GetOpTypeName(op), attrName.strref());
    if (!attrType) { continue; }

    llvm::SmallVector<uint8_t, 4> packedAttrData;
    StringRef strAttrType;
    auto attrValue = kv.getValue();
    if (attrType.value() == ::oneflow::kAtInt32) {
      strAttrType = "i32";
      packedAttrData.resize(4);
      // memcpy();
    } else if (attrType.value() == ::oneflow::kAtInt64) {
      strAttrType = "i64";

    } else if (attrType.value() == ::oneflow::kAtBool) {
      strAttrType = "bool";

    } else if (attrType.value() == ::oneflow::kAtFloat) {
      strAttrType = "f32";

    } else if (attrType.value() == ::oneflow::kAtDouble) {
      strAttrType = "f64";

    } else if (attrType.value() == ::oneflow::kAtString) {
      strAttrType = "str";

    } else if (attrType.value() == ::oneflow::kAtShape) {
      strAttrType = "shape";

    } else if (attrType.value() == ::oneflow::kAtStride) {
      strAttrType = "stride";

    } else if (attrType.value() == ::oneflow::kAtDataType) {
      strAttrType = "dtype";

    } else if (attrType.value() == ::oneflow::kAtListInt32) {
      strAttrType = "i32s";

    } else if (attrType.value() == ::oneflow::kAtListInt64) {
      strAttrType = "i64s";

    } else if (attrType.value() == ::oneflow::kAtListFloat) {
      strAttrType = "f32s";

    } else if (attrType.value() == ::oneflow::kAtListDataType) {
      strAttrType = "dtypes";

    } else if (attrType.value() == ::oneflow::kAtListShape) {
      strAttrType = "shapes";

    } else if (attrType.value() == ::oneflow::kAtListStride) {
      strAttrType = "strides";

    } else if (attrType.value() == ::oneflow::kAtListString) {
      strAttrType = "strs";
    } else {
      llvm::errs() << "error attribute type: " << attrType.value() << "\n";
      exit(1);
    }
    oneflow_lite_AttrDef_start(builder);
    oneflow_lite_AttrDef_type_add(builder, builder.createString(strAttrType));
    oneflow_lite_AttrDef_key_add(builder, builder.createString(attrName.strref()));
    attrDefs.push_back(oneflow_lite_AttrDef_end(builder));
  }
  return builder.createOffsetVecDestructive(attrDefs);
}

static flatbuffers_vec_ref_t createLiteOpAttrs(FlatbufferBuilder& builder, Operation* op) {
  if (llvm::dyn_cast<VariableOp>(op)) {
    llvm::SmallVector<oneflow_lite_AttrDef_ref_t, 4> attrDefs;
    return builder.createOffsetVecDestructive(attrDefs);
  }
  return createLiteUserOpAttrs(builder, op);
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
  oneflow_lite_TensorDef_type_add(
      builder, builder.createString(getLiteStringElementType(type.getElementType())));
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
  pm.addPass(createCanonicalizerPass());

  LiteBufferStrategy bufferStrategy;
  pm.addPass(createLiteMemoryPlanningPass(&bufferStrategy));
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Failed to run oneflow lite compilation passes.\n";
    return failure();
  }

  Operation* root = nullptr;
  module.getOperation()->walk([&](oneflow::Job job) -> WalkResult {
    root = job.getOperation();
    return WalkResult::interrupt();
  });
  if (!root) {
    llvm::errs() << "Job not found in module: " << *module;
    return failure();
  }
  llvm::errs() << *module << "\n";

  auto funcName = root->getAttrOfType<StringAttr>("sym_name");
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

  root->walk([&](Operation* op) {
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
