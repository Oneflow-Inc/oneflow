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

#include "OneFlow/Transform/Lowering/LoweringAscendUtils.h"
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

class AscendValue {
 public:
  AscendValue() = default;
  AscendValue(const std::shared_ptr<ge::Operator>& op, const ge::TensorDesc& type,
              StringRef componentName)
      : op_(op), type_(type), componentName_(componentName), componentIndex_(-1) {}
  AscendValue(const std::shared_ptr<ge::Operator>& op, const ge::TensorDesc& type,
              StringRef componentName, int componentIndex)
      : op_(op), type_(type), componentName_(componentName), componentIndex_(componentIndex) {}

  AscendValue(const AscendValue&) = default;

  const std::shared_ptr<ge::Operator>& getOperation() const { return op_; }

  const ge::TensorDesc& getType() const { return type_; }

  StringRef getComponentName() const { return componentName_; }
  int getComponentIndex() const { return componentIndex_; }

  StringRef getComponentNameAndIndex() const {
    if (componentIndex_ < 0) { return componentName_; }
    auto name = componentName_ + llvm::Twine(componentIndex_);
    return StringRef(name.str());
  }

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

class AscendCompiler {
 public:
  AscendCompiler() = default;

  void addInputs(llvm::SmallVector<Value, 4>& operands);

  void lowerOp(VariableOp op, StringRef checkpointDir);
  void lowerOp(Conv2DOp op);
  void lowerOp(NormalizationInferenceOp op);
  void lowerOp(ReluOp op);
  void lowerOp(MaxPool2DOp op);
  void lowerOp(AvgPool2DOp op);
  void lowerOp(Add2Op op);
  void lowerOp(AdaptiveAvgPool2DOp op);
  void lowerOp(MatmulOp op);
  void lowerOp(BroadcastAddOp op);
  void lowerOp(ReshapeOp op);
  void lowerOp(func::ReturnOp op);

  void serializeToBuffer(llvm::SmallVector<uint8_t, 4>* data);

 private:
  AscendValue getValue(Value value) const {
    auto it = ascendVals.find(value);
    assert(it != ascendVals.end());
    return it->second;
  }

  template<typename T>
  std::shared_ptr<T> createOp(llvm::Twine opName) {
    auto op = std::make_shared<T>(opName.str());
    ascendOps.push_back(op);
    return op;
  }

  llvm::SmallVector<AscendValue, 4> inputs;
  llvm::SmallVector<AscendValue, 4> results;
  llvm::SmallVector<std::shared_ptr<ge::Operator>, 4> ascendOps;
  llvm::DenseMap<Value, AscendValue> ascendVals;
};

void AscendCompiler::serializeToBuffer(llvm::SmallVector<uint8_t, 4>* data) {
  std::vector<ge::Operator> ins;
  std::vector<std::pair<ge::Operator, ge::AscendString>> outs;
  for (auto in : inputs) { ins.push_back(*(in.getOperation())); }
  for (auto out : results) {
    outs.push_back(std::make_pair(*(out.getOperation()), out.getComponentNameAndIndex().data()));
  }
  ge::Graph graph("ascend-graph");
  graph.SetInputs(ins).SetOutputs(outs);

  if (!graph.IsValid()) {
    llvm::errs() << "ascend graph is invalid\n";
    exit(1);
  }
  const char* outputFilename = ".__TMP__ascend_graph";
  graph.SaveToFile(outputFilename);

  std::string errorMessage;
  auto f = mlir::openInputFile(outputFilename, &errorMessage);
  if (!f) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }
  data->resize(f->getBufferSize());
  memcpy(data->data(), f->getBufferStart(), data->size());

  // clean temp file
  if (0 != remove(outputFilename)) {
    llvm::errs() << "faile to clean temp file\n";
    exit(1);
  }
}

void AscendCompiler::addInputs(llvm::SmallVector<Value, 4>& operands) {
  for (auto operand : llvm::enumerate(operands)) {
    llvm::Twine opName = "input_" + llvm::Twine(operand.index());
    auto inputOp = createOp<ge::op::Data>(opName.str());
    auto ascendType = convertAscendType(operand.value().getType());
    inputOp->update_input_desc_x(ascendType);
    inputOp->update_output_desc_y(ascendType);
    inputs.push_back(AscendValue(inputOp, ascendType, "y"));
    ascendVals[operand.value()] = inputs.back();
  }
}

void AscendCompiler::lowerOp(VariableOp op, StringRef checkpointDir) {
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
  ascendVals[op.output()] = AscendValue(constantOp, ascendType, "y");
}

#define SET_INPUT(op, name, value) \
  op->set_input_##name##_by_name(*(value.getOperation()), value.getComponentNameAndIndex().data())

void AscendCompiler::lowerOp(Conv2DOp op) {
  auto conv2DOp = createOp<ge::op::Conv2D>(op.op_name());
  conv2DOp->set_attr_pads(convertPaddings(op.padding_before()));
  conv2DOp->set_attr_dilations(convertDilations(op.dilation_rate()));
  conv2DOp->set_attr_strides(convertStrides(op.strides()));
  conv2DOp->set_attr_groups(op.groups());
  conv2DOp->set_attr_data_format(convertDataFormat(op.data_format()).data());

  SET_INPUT(conv2DOp, x, getValue(op.in()));
  SET_INPUT(conv2DOp, filter, getValue(op.weight()));
  if (op.bias()) { SET_INPUT(conv2DOp, bias, getValue(op.bias())); }
  auto outType = convertAscendType(op.out().getType());
  conv2DOp->update_output_desc_y(outType);

  auto output = AscendValue(conv2DOp, outType, "y");

  if (op._add_to_output()) {
    auto addOp = createOp<ge::op::AddV2>(op.op_name() + "_add_to_output");
    SET_INPUT(addOp, x1, output);
    SET_INPUT(addOp, x2, getValue(op._add_to_output()));
    addOp->update_output_desc_y(outType);
    output = AscendValue(addOp, outType, "y");
  }
  ascendVals[op.out()] = output;
}

void AscendCompiler::lowerOp(NormalizationInferenceOp op) {
  auto batchNormOp = createOp<ge::op::BNInfer>(op.op_name());
  batchNormOp->set_attr_epsilon(op.epsilon().convertToFloat());

  SET_INPUT(batchNormOp, x, getValue(op.x()));
  SET_INPUT(batchNormOp, mean, getValue(op.moving_mean()));
  SET_INPUT(batchNormOp, variance, getValue(op.moving_variance()));
  SET_INPUT(batchNormOp, scale, getValue(op.gamma()));
  SET_INPUT(batchNormOp, offset, getValue(op.beta()));

  auto outType = convertAscendType(op.y().getType());
  batchNormOp->update_output_desc_y(outType);

  auto output = AscendValue(batchNormOp, outType, "y");
  if (op._add_to_output()) {
    auto addOp = createOp<ge::op::AddV2>(op.op_name() + "_add_to_output");
    SET_INPUT(addOp, x1, output);
    SET_INPUT(addOp, x2, getValue(op._add_to_output()));
    addOp->update_output_desc_y(outType);
    output = AscendValue(addOp, outType, "y");
  }
  ascendVals[op.y()] = output;
}

void AscendCompiler::lowerOp(ReluOp op) {
  auto reluOp = createOp<ge::op::Relu>(op.op_name());
  SET_INPUT(reluOp, x, getValue(op.x()));
  auto outType = convertAscendType(op.y().getType());
  reluOp->update_output_desc_y(outType);
  ascendVals[op.y()] = AscendValue(reluOp, outType, "y");
}

void AscendCompiler::lowerOp(MaxPool2DOp op) {
  auto maxPoolOp = createOp<ge::op::MaxPoolV3>(op.op_name());
  maxPoolOp->set_attr_ksize(convertKernelSize(op.kernel_size()));
  maxPoolOp->set_attr_pads(convertPaddings(op.padding()));
  maxPoolOp->set_attr_strides(convertStrides(op.stride()));
  maxPoolOp->set_attr_ceil_mode(op.ceil_mode());
  maxPoolOp->set_attr_padding_mode("CALCULATED");
  maxPoolOp->set_attr_global_pooling(false);

  SET_INPUT(maxPoolOp, x, getValue(op.x()));
  auto outType = convertAscendType(op.y().getType());
  maxPoolOp->update_output_desc_y(outType);
  ascendVals[op.y()] = AscendValue(maxPoolOp, outType, "y");
}

void AscendCompiler::lowerOp(AvgPool2DOp op) {
  auto avgPoolOp = createOp<ge::op::AvgPoolV2>(op.op_name());
  avgPoolOp->set_attr_ksize(convertKernelSize(op.kernel_size()));
  avgPoolOp->set_attr_pads(convertPaddings(op.padding()));
  avgPoolOp->set_attr_strides(convertStrides(op.stride()));
  avgPoolOp->set_attr_ceil_mode(op.ceil_mode());
  avgPoolOp->set_attr_padding_mode("CALCULATED");
  avgPoolOp->set_attr_global_pooling(false);
  avgPoolOp->set_attr_exclusive(!op.count_include_pad());

  SET_INPUT(avgPoolOp, x, getValue(op.x()));
  auto outType = convertAscendType(op.y().getType());
  avgPoolOp->update_output_desc_y(outType);
  ascendVals[op.y()] = AscendValue(avgPoolOp, outType, "y");
}

void AscendCompiler::lowerOp(Add2Op op) {
  auto addOp = createOp<ge::op::AddV2>(op.op_name());
  SET_INPUT(addOp, x1, getValue(op.in0()));
  SET_INPUT(addOp, x2, getValue(op.in1()));
  auto outType = convertAscendType(op.out().getType());
  addOp->update_output_desc_y(outType);
  ascendVals[op.out()] = AscendValue(addOp, outType, "y");
}

void AscendCompiler::lowerOp(AdaptiveAvgPool2DOp op) {
  auto adaptiveAvgPoolOp = createOp<ge::op::AdaptiveAvgPool2d>(op.op_name());
  ArrayAttr output_size = op.output_size();
  assert(output_size.size() == 2);
  int64_t s0 = output_size[0].dyn_cast<IntegerAttr>().getSInt();
  int64_t s1 = output_size[1].dyn_cast<IntegerAttr>().getSInt();
  adaptiveAvgPoolOp->set_attr_output_size(ge::Operator::OpListInt({s0, s1}));
  SET_INPUT(adaptiveAvgPoolOp, x, getValue(op.x()));
  auto outType = convertAscendType(op.y().getType());
  adaptiveAvgPoolOp->update_output_desc_y(outType);
  ascendVals[op.y()] = AscendValue(adaptiveAvgPoolOp, outType, "y");
}

void AscendCompiler::lowerOp(MatmulOp op) {
  auto matmulOp = createOp<ge::op::MatMulV2>(op.op_name());
  matmulOp->set_attr_transpose_x1(op.transpose_a());
  matmulOp->set_attr_transpose_x2(op.transpose_b());

  SET_INPUT(matmulOp, x1, getValue(op.a()));
  SET_INPUT(matmulOp, x2, getValue(op.b()));
  auto outType = convertAscendType(op.out().getType());
  matmulOp->update_output_desc_y(outType);

  auto output = AscendValue(matmulOp, outType, "y");
  if (op._add_to_output()) {
    auto addOp = createOp<ge::op::AddV2>(op.op_name() + "_add_to_output");
    SET_INPUT(addOp, x1, output);
    SET_INPUT(addOp, x2, getValue(op._add_to_output()));
    addOp->update_output_desc_y(outType);
    output = AscendValue(addOp, outType, "y");
  }
  ascendVals[op.out()] = output;
}

void AscendCompiler::lowerOp(BroadcastAddOp op) {
  auto addOp = createOp<ge::op::AddV2>(op.op_name());
  SET_INPUT(addOp, x1, getValue(op.x()));
  SET_INPUT(addOp, x2, getValue(op.y()));
  auto outType = convertAscendType(op.z().getType());
  addOp->update_output_desc_y(outType);
  ascendVals[op.z()] = AscendValue(addOp, outType, "y");
}

void AscendCompiler::lowerOp(ReshapeOp op) {
  llvm::SmallVector<int64_t, 4> shape;
  for (auto v : op.shape()) { shape.push_back(v.dyn_cast<IntegerAttr>().getSInt()); }
  auto constantOp = createOp<ge::op::Const>(op.op_name() + "_shape");
  auto shapeType =
      ge::TensorDesc(ge::Shape(std::vector<int64_t>{static_cast<int64_t>(shape.size())}),
                     ge::FORMAT_NCHW, ge::DT_INT64);
  auto tensor = std::make_shared<ge::Tensor>();
  tensor->SetTensorDesc(shapeType);
  tensor->SetData(reinterpret_cast<const uint8_t*>(shape.data()), shape.size() * sizeof(int64_t));
  constantOp->set_attr_value(*tensor);

  auto reshapeOp = createOp<ge::op::Reshape>(op.op_name());
  SET_INPUT(reshapeOp, x, getValue(op.in()));
  SET_INPUT(reshapeOp, shape, (AscendValue(constantOp, shapeType, "y")));

  auto outType = convertAscendType(op.out().getType());
  reshapeOp->update_output_desc_y(outType);
  ascendVals[op.out()] = AscendValue(reshapeOp, outType, "y");
}

void AscendCompiler::lowerOp(func::ReturnOp op) {
  for (auto operand : op.operands()) { results.push_back(getValue(operand)); }
}

#undef SET_INPUT

LogicalResult loweringAscend(OpBuilder& builder, Operation* callee, StringRef checkpointDir,
                             llvm::SmallVector<uint8_t, 4>* loweringData) {
  AscendCompiler compiler;
  llvm::SmallVector<Value, 4> inputs;
  auto func = dyn_cast<func::FuncOp>(callee);
  for (auto argument : func.getArguments()) { inputs.push_back(argument); }

  compiler.addInputs(inputs);

  func.getBody().walk([&](Operation* op) {
    if (auto x = dyn_cast<VariableOp>(op)) {
      compiler.lowerOp(x, checkpointDir);
    } else if (auto x = dyn_cast<Conv2DOp>(op)) {
      compiler.lowerOp(x);
    } else if (auto x = dyn_cast<NormalizationInferenceOp>(op)) {
      compiler.lowerOp(x);
    } else if (auto x = dyn_cast<ReluOp>(op)) {
      compiler.lowerOp(x);
    } else if (auto x = dyn_cast<MaxPool2DOp>(op)) {
      compiler.lowerOp(x);
    } else if (auto x = dyn_cast<AvgPool2DOp>(op)) {
      compiler.lowerOp(x);
    } else if (auto x = dyn_cast<Add2Op>(op)) {
      compiler.lowerOp(x);
    } else if (auto x = dyn_cast<AdaptiveAvgPool2DOp>(op)) {
      compiler.lowerOp(x);
    } else if (auto x = dyn_cast<MatmulOp>(op)) {
      compiler.lowerOp(x);
    } else if (auto x = dyn_cast<BroadcastAddOp>(op)) {
      compiler.lowerOp(x);
    } else if (auto x = dyn_cast<ReshapeOp>(op)) {
      compiler.lowerOp(x);
    } else if (auto x = dyn_cast<func::ReturnOp>(op)) {
      compiler.lowerOp(x);
    } else {
      llvm::errs() << "could not lowerring " << op->getName() << " for backend ascend\n";
      exit(1);
    }
  });
  compiler.serializeToBuffer(loweringData);
  return success();
}

}  // namespace lite
}  // namespace oneflow
}  // namespace mlir
