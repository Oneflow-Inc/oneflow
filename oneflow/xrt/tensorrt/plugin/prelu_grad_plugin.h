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
#ifndef ONEFLOW_XRT_TENSORRT_PLUGIN_PRELU_GRAD_PLUGIN_H_
#define ONEFLOW_XRT_TENSORRT_PLUGIN_PRELU_GRAD_PLUGIN_H_

#include "oneflow/xrt/tensorrt/trt_plugin.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class PreluGradPlugin : public TrtPlugin {
 public:
  PreluGradPlugin(const std::string& name, bool alpha_requires_grad)
      : name_(name), alpha_requires_grad_(alpha_requires_grad) {}

  const char* getPluginType() const TRT_NOEXCEPT override { return "PReluGrad"; }

  int getNbOutputs() const TRT_NOEXCEPT { return 2; }

  nvinfer1::DimsExprs getOutputDimensions(int output_index, const nvinfer1::DimsExprs* inputs,
                                          int nb_inputs, nvinfer1::IExprBuilder& expr_builder)
      TRT_NOEXCEPT override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nb_outputs) const TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
                                 int nb_outputs) TRT_NOEXCEPT override;

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
              const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
              void* const* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* input_types,
                                       int nb_inputs) const TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }

  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override;

 private:
  std::string name_;
  bool alpha_requires_grad_;
};

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_PLUGIN_PRELU_GRAD_PLUGIN_H_
