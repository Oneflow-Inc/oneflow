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
#ifndef ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_TRANSFORM_LOWERING_LOWERINGASCENDUTILS_H_
#define ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_TRANSFORM_LOWERING_LOWERINGASCENDUTILS_H_

#include <vector>

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"
#include "OneFlow/OneFlowLiteUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

// huawei ascend sdk headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include "op_proto/built-in/inc/all_ops.h"
#pragma GCC diagnostic pop

namespace mlir {
namespace oneflow {
namespace lite {

inline ge::Shape convertAscendShape(ArrayRef<int64_t> shape) {
  return ge::Shape(std::vector<int64_t>{shape.begin(), shape.end()});
}

inline Optional<ge::DataType> convertAscendElementType(Type type) {
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

inline Optional<ge::DataType> convertAscendElementType(::mlir::oneflow::DataType type) {
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

inline ge::TensorDesc convertAscendType(Type type) {
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

inline ge::TensorDesc convertAscendType(::mlir::oneflow::DataType type, ArrayRef<int64_t> shape) {
  auto elementType = convertAscendElementType(type);
  if (!elementType) {
    llvm::errs() << "element type " << static_cast<uint32_t>(type) << " is not supported\n";
    exit(1);
  }
  return ge::TensorDesc(convertAscendShape(shape), ge::FORMAT_NCHW, elementType.value());
}

inline ge::TensorDesc convertAscendType(Attribute type, Attribute shape) {
  SmallVector<int64_t, 4> shapeArray;
  for (auto v : shape.dyn_cast<ArrayAttr>().getValue()) {
    shapeArray.push_back(v.dyn_cast<IntegerAttr>().getSInt());
  }
  return convertAscendType(type.dyn_cast<mlir::oneflow::DataTypeAttr>().getValue(), shapeArray);
}

inline ge::Operator::OpListInt convertPaddings(ArrayAttr paddings) {
  assert(paddings.size() == 2 || paddings.size() == 4);
  if (paddings.size() == 2) {
    int s0 = paddings[0].dyn_cast<IntegerAttr>().getSInt();
    int s1 = paddings[1].dyn_cast<IntegerAttr>().getSInt();
    return ge::Operator::OpListInt({s0, s0, s1, s1});
  } else {
    int s0 = paddings[0].dyn_cast<IntegerAttr>().getSInt();
    int s1 = paddings[1].dyn_cast<IntegerAttr>().getSInt();
    int s2 = paddings[2].dyn_cast<IntegerAttr>().getSInt();
    int s3 = paddings[3].dyn_cast<IntegerAttr>().getSInt();
    return ge::Operator::OpListInt({s0, s1, s2, s3});
  }
}

inline ge::Operator::OpListInt convertStrides(ArrayAttr strides) {
  assert(strides.size() == 2);
  int s0 = strides[0].dyn_cast<IntegerAttr>().getSInt();
  int s1 = strides[1].dyn_cast<IntegerAttr>().getSInt();
  return ge::Operator::OpListInt({1, 1, s0, s1});
}

inline ge::Operator::OpListInt convertDilations(ArrayAttr dilations) {
  return convertStrides(dilations);
}

inline ge::Operator::OpListInt convertKernelSize(ArrayAttr kernel_size) {
  return convertStrides(kernel_size);
}

inline StringRef convertDataFormat(StringRef dataFormat) {
  if (dataFormat == "nchw" || dataFormat == "NCHW" || dataFormat == "channels_first") {
    return StringRef("NCHW");
  } else if (dataFormat == "nhwc" || dataFormat == "NHWC" || dataFormat == "channels_last") {
    return StringRef("NHWC");
  } else {
    llvm::errs() << "unsupport data format " << dataFormat << "\n";
    exit(1);
  }
}

}  // namespace lite
}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_TRANSFORM_LOWERING_LOWERINGASCENDUTILS_H_
