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
#ifndef ONEFLOW_XRT_TENSORRT_TRT_PLUGIN_H_
#define ONEFLOW_XRT_TENSORRT_TRT_PLUGIN_H_

#include "NvInfer.h"
#include "oneflow/core/common/util.h"

#define TRT_NOEXCEPT noexcept

namespace oneflow {
namespace xrt {
namespace tensorrt {

class TrtPlugin : public nvinfer1::IPluginV2DynamicExt {
 public:
  TrtPlugin() = default;
  TrtPlugin(const void* serialized_data, size_t length) {}

  virtual ~TrtPlugin() = default;

  virtual const char* getPluginType() const TRT_NOEXCEPT = 0;
  virtual const char* getPluginVersion() const TRT_NOEXCEPT { return "v2"; }

  virtual int initialize() TRT_NOEXCEPT { return 0; }
  virtual void terminate() TRT_NOEXCEPT {}
  virtual void destroy() TRT_NOEXCEPT = 0;

  virtual nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT = 0;

  virtual int getNbOutputs() const TRT_NOEXCEPT = 0;

  virtual nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT = 0;

  virtual bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* in_out,
                                         int nb_inputs, int nb_outputs) TRT_NOEXCEPT = 0;

  virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nb_inputs,
                               const nvinfer1::DynamicPluginTensorDesc* out,
                               int nb_outputs) TRT_NOEXCEPT {}

  virtual size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nb_inputs,
                                  const nvinfer1::PluginTensorDesc* outputs,
                                  int nb_outputs) const TRT_NOEXCEPT {
    return 0;
  }

  virtual int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                      const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
                      void* const* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT = 0;

  virtual nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* input_types,
                                               int nb_inputs) const TRT_NOEXCEPT = 0;

  void setPluginNamespace(const char* plugin_namespace) TRT_NOEXCEPT override {
    namespace_ = plugin_namespace;
  }

  const char* getPluginNamespace() const TRT_NOEXCEPT override { return namespace_.c_str(); }

  virtual size_t getSerializationSize() const TRT_NOEXCEPT {
    UNIMPLEMENTED();
    return 0;
  }
  virtual void serialize(void* buffer) const TRT_NOEXCEPT { UNIMPLEMENTED(); }

 private:
  std::string namespace_;
};

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_TRT_PLUGIN_H_
