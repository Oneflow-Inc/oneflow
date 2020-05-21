#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/trt_value.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

nvinfer1::ITensor *TrtOpContext::Input(const std::string &name) {
  return Input(ArgumentFromKey(name));
}

nvinfer1::ITensor *TrtOpContext::Output(const std::string &name) {
  return Output(ArgumentFromKey(name));
}

nvinfer1::ITensor *TrtOpContext::Input(const Argument &arg) {
  CHECK_GT(param_.inputs.count(arg), 0);
  return param_.inputs.at(arg).AsTensor(builder());
}

nvinfer1::ITensor *TrtOpContext::Output(const Argument &arg) {
  CHECK_GT(outputs_.count(arg), 0);
  return outputs_.at(arg).AsTensor(builder());
}

nvinfer1::Weights &TrtOpContext::Weight(const std::string &name) {
  return Weight(ArgumentFromKey(name));
}

nvinfer1::Weights &TrtOpContext::Weight(const Argument &arg) {
  CHECK_GT(param_.inputs.count(arg), 0);
  return param_.inputs.at(arg).AsWeight(builder());
}

void TrtOpContext::SetOutput(const std::string &name, nvinfer1::ITensor *tensor) {
  SetOutput(name, TrtValue::Tensor(builder(), tensor));
}

void TrtOpContext::SetOutput(const std::string &name, const TrtValue &value) {
  Argument arg = ArgumentFromKey(name);
  outputs_[arg] = value;
  nvinfer1::ITensor *tensor = builder()->GetTensor(value.handle());
  tensor->setName(arg.name().c_str());
}

DataType TrtOpContext::InputType(const std::string &name) const {
  return ArgumentFromKey(name).data_type();
}

DataType TrtOpContext::OutputType(const std::string &name) const {
  return ArgumentFromKey(name).data_type();
}

Shape TrtOpContext::InputShape(const std::string &name) const {
  return ArgumentFromKey(name).shape();
}

Shape TrtOpContext::OutputShape(const std::string &name) const {
  return ArgumentFromKey(name).shape();
}

Argument TrtOpContext::ArgumentFromKey(const std::string &key) const {
  CHECK_GT(param_.arguments.count(key), 0);
  return param_.arguments.at(key);
}

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
