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
#ifndef ONEFLOW_XRT_TENSORRT_OPS_OP_CONTEXT_H_
#define ONEFLOW_XRT_TENSORRT_OPS_OP_CONTEXT_H_

#include "NvInfer.h"

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/xrt/argument.h"
#include "oneflow/xrt/kernel/op_context.h"
#include "oneflow/xrt/tensorrt/trt_builder.h"
#include "oneflow/xrt/tensorrt/trt_value.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/stl.h"
#include "oneflow/xrt/xrt.pb.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class TrtOpContext : public OpContext {
 public:
  struct Param {
    std::string op_name;

    TrtBuilder *builder;
    // Config proto related to the operator
    const PbMessage *message;
    // Input operands
    util::Map<Argument, TrtValue> inputs;
    std::vector<std::string> output_names;
    int num_outputs;

    util::Map<std::string, Argument> arguments;
  };

  explicit TrtOpContext(const Param &param) : OpContext(*param.message), param_(param) {}

  virtual ~TrtOpContext() = default;

  const Param &param() const { return param_; }

  TrtBuilder *builder() const { return param_.builder; }

  const std::string &op_name() const { return param_.op_name; }
  
  const std::string &SoleOutputName() const; 

  // Return input named `name` as tensor
  nvinfer1::ITensor *Input(const std::string &name);
  nvinfer1::ITensor *Input(const Argument &arg);
  nvinfer1::ITensor *SoleInput();
  // Return output named `name` as tensor
  nvinfer1::ITensor *Output(const std::string &name);
  nvinfer1::ITensor *Output(const Argument &arg);
  nvinfer1::ITensor *SoleOutput();

  // Return weight named `name` as weight
  nvinfer1::Weights &Weight(const std::string &name);
  nvinfer1::Weights &Weight(const Argument &arg);

  int num_inputs() const { return param_.inputs.size(); }
  int num_outputs() const { return param_.num_outputs; }
  // Return inputs as TrtValues
  const util::Map<Argument, TrtValue> &inputs() const { return param_.inputs; }
  // Return output as TrtValues
  const util::Map<Argument, TrtValue> &outputs() const { return outputs_; }

  // Setup the output `output_name` with XlaOp
  void SetOutput(const std::string &name, nvinfer1::ITensor *tensor);
  // Setup the output `output_name` with TrtValue
  void SetOutput(const std::string &name, const TrtValue &value);
  void SetSoleOutput(nvinfer1::ITensor *tensor);

  // Return input `name` shape as Shape
  Shape InputShape(const std::string &name) const;
  Shape SoleInputShape() const;
  // Return output `name` shape as Shape
  Shape OutputShape(const std::string &name) const;
  Shape SoleOutputShape() const;  

  // Input data type
  DataType InputType(const std::string &name) const;
  DataType SoleInputType() const;
  // Output data type
  DataType OutputType(const std::string &name) const;
  DataType SoleOutputType() const;

  bool HasInput(const std::string &name) const;

 private:
  TrtOpContext() = delete;
  Argument ArgumentFromKey(const std::string &key) const;

  Param param_;
  // Output operands
  util::Map<Argument, TrtValue> outputs_;
};

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_OPS_OP_CONTEXT_H_
