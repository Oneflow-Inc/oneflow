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
#include "OneFlow/Transform/Lowering/LoweringAscend.h"

#include <memory>
#include <vector>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"
#include "OneFlow/OneFlowLiteUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"

// huawei ascend sdk headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include "op_proto/built-in/inc/all_ops.h"
#pragma GCC diagnostic pop

namespace mlir {
namespace oneflow {
namespace lite {

static ge::Shape convertAscendShape(ArrayRef<int64_t> shape) {
  return ge::Shape(std::vector<int64_t>{shape.begin(), shape.end()});
}

static Optional<ge::DataType> convertAscendElementType(Type type) {
  assert(type.isIntOrFloat());
  if (type.isF16()) {
    return ge::DT_FLOAT16;
  } else if (type.isF32()) {
    return ge::DT_FLOAT;
  } else if (type.isF64()) {
    return ge::DT_DOUBLE;
  } else if (type.isSignedInteger()) {
    int bitwidth = type.getIntOrFloatBitWidth();
    if (bitwidth == 8) {
      return ge::DT_INT8;
    } else if (bitwidth == 16) {
      return ge::DT_INT16;
    } else if (bitwidth == 32) {
      return ge::DT_INT32;
    } else if (bitwidth == 64) {
      return ge::DT_INT64;
    } else {
      return llvm::None;
    }
  } else if (type.isUnsignedInteger()) {
    int bitwidth = type.getIntOrFloatBitWidth();
    if (bitwidth == 8) {
      return ge::DT_UINT8;
    } else if (bitwidth == 16) {
      return ge::DT_UINT16;
    } else if (bitwidth == 32) {
      return ge::DT_UINT32;
    } else if (bitwidth == 64) {
      return ge::DT_UINT64;
    } else {
      return llvm::None;
    }
  } else {
    return llvm::None;
  }
}

static Optional<ge::DataType> convertAscendElementType(::mlir::oneflow::DataType type) {
  switch (type) {
    case ::mlir::oneflow::DataType::DT_Bool: return ge::DT_BOOL;
    case ::mlir::oneflow::DataType::DT_Char: return ge::DT_UINT8;
    case ::mlir::oneflow::DataType::DT_Float16: return ge::DT_FLOAT16;
    case ::mlir::oneflow::DataType::DT_Float: return ge::DT_FLOAT;
    case ::mlir::oneflow::DataType::DT_Double: return ge::DT_DOUBLE;
    case ::mlir::oneflow::DataType::DT_Int8: return ge::DT_INT8;
    case ::mlir::oneflow::DataType::DT_Int32: return ge::DT_INT32;
    case ::mlir::oneflow::DataType::DT_Int64: return ge::DT_INT64;
    case ::mlir::oneflow::DataType::DT_UInt8: return ge::DT_UINT8;
    default: {
      return llvm::None;
    }
  }
}

static ge::TensorDesc convertAscendType(Type type) {
  auto tensorType = type.cast<TensorType>();
  assert(tensorType && "type should be tensor type");
  auto elementType = convertAscendElementType(tensorType.getElementType());
  if (!elementType) {
    llvm::errs() << "element type " << tensorType.getElementType() << " is not supported\n";
    exit(1);
  }
  return ge::TensorDesc(convertAscendShape(tensorType.getShape()), ge::FORMAT_NCHW,
                        elementType.value());
}

static ge::TensorDesc convertAscendType(::mlir::oneflow::DataType type, ArrayRef<int64_t> shape) {
  auto elementType = convertAscendElementType(type);
  if (!elementType) {
    llvm::errs() << "element type " << static_cast<uint32_t>(type) << " is not supported\n";
    exit(1);
  }
  return ge::TensorDesc(convertAscendShape(shape), ge::FORMAT_NCHW, elementType.value());
}

static ge::TensorDesc convertAscendType(Attribute type, Attribute shape) {
  SmallVector<int64_t, 4> shapeArray;
  for (auto v : shape.dyn_cast<ArrayAttr>().getValue()) {
    shapeArray.push_back(v.dyn_cast<IntegerAttr>().getSInt());
  }
  return convertAscendType(type.dyn_cast<mlir::oneflow::DataTypeAttr>().getValue(), shapeArray);
}

class AscendValue {
 public:
  AscendValue() = default;
  AscendValue(const std::shared_ptr<ge::Operator>& op, const ge::TensorDesc& type,
              StringRef componentName)
      : op_(op), type_(type), componentName_(componentName), componentIndex_(0) {}
  AscendValue(const std::shared_ptr<ge::Operator>& op, const ge::TensorDesc& type,
              StringRef componentName, int componentIndex)
      : op_(op), type_(type), componentName_(componentName), componentIndex_(componentIndex) {}

  AscendValue(const AscendValue&) = default;

  const std::shared_ptr<ge::Operator>& getOperation() const { return op_; }

  const ge::TensorDesc& getType() const { return type_; }

  StringRef getComponentName() const { return componentName_; }
  int getComponentIndex() const { return componentIndex_; }

  void setOperation(const std::shared_ptr<ge::Operator>& op) { op_ = op; }
  void setType(const ge::TensorDesc& type) { type_ = type; }
  void setComponentName(StringRef componentName) { componentName_ = componentName; }
  void setComponentIndex(int componentIndex) { componentIndex_ = componentIndex; }

 private:
  std::shared_ptr<ge::Operator> op_;
  ge::TensorDesc type_;
  StringRef componentName_;
  int componentIndex_;
};

class LoweringAscendContext {
 public:
  LoweringAscendContext() = default;

  void addInputs(llvm::SmallVector<Value, 4>& operands);
  void addOutputs(llvm::SmallVector<Value, 4>& results);

  void loweringVariableOp(VariableOp op, StringRef checkpointDir);
  void loweringConv2DOp(Conv2DOp op);

  template<typename T>
  std::shared_ptr<T> createOp(StringRef opName) {
    auto op = std::make_shared<T>(opName.str());
    ascendOps.push_back(op);
    return op;
  }

 private:
  llvm::SmallVector<std::shared_ptr<ge::Operator>, 4> ascendOps;
  llvm::DenseMap<Value, AscendValue> ascendVals;
};

void LoweringAscendContext::addInputs(llvm::SmallVector<Value, 4>& operands) {
  for (auto operand : llvm::enumerate(operands)) {
    llvm::Twine opName = "argument_" + llvm::Twine(operand.index());
    auto inputOp = createOp<ge::op::Data>(opName.str());
    auto ascendType = convertAscendType(operand.value().getType());
    inputOp->update_input_desc_x(ascendType);
    inputOp->update_output_desc_y(ascendType);
    ascendVals[operand.value()] = AscendValue(inputOp, ascendType, "y", 0);
  }
}

void LoweringAscendContext::loweringVariableOp(VariableOp op, StringRef checkpointDir) {
  auto ascendType = convertAscendType(op.data_typeAttr(), op.shapeAttr());
  llvm::SmallString<128> inputFilename;
  llvm::sys::path::native(checkpointDir + "/" + op.op_name() + "/out", inputFilename);
  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }
  auto constantOp = createOp<ge::op::Const>(op.op_name());
  auto tensor = std::make_shared<ge::Tensor>();
  tensor->SetTensorDesc(ascendType);
  tensor->SetData(reinterpret_cast<const uint8_t*>(input->getBufferStart()),
                  input->getBufferSize());
  constantOp->set_attr_value(*tensor);
  ascendVals[op.output()] = AscendValue(constantOp, ascendType, "y", 0);
}

void LoweringAscendContext::loweringConv2DOp(Conv2DOp op) {}

LogicalResult loweringAscend(OpBuilder& builder, Operation* callee, StringRef checkpointDir,
                             llvm::SmallVector<uint8_t, 4>* loweringData) {
  llvm::SmallVector<Value, 4> inputs;
  auto func = dyn_cast<oneflow::Job>(callee);
  for (auto argument : func.getArguments()) { inputs.push_back(argument); }

  LoweringAscendContext loweringContext;
  loweringContext.addInputs(inputs);

  callee->walk([&](Operation* op) {
    if (auto varOp = dyn_cast<VariableOp>(op)) {
      loweringContext.loweringVariableOp(varOp, checkpointDir);
    } else if (auto convOp = dyn_cast<Conv2DOp>(op)) {
      loweringContext.loweringConv2DOp(convOp);
    } else {
      llvm::errs() << "could not lowerring " << op->getName() << " for backend ascend\n";
      exit(1);
    }
  });
  return success();
}

}  // namespace lite
}  // namespace oneflow
}  // namespace mlir
