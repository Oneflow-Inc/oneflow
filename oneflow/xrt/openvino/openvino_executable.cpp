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
#include "oneflow/xrt/openvino/openvino_executable.h"
#include "oneflow/xrt/platform.h"

namespace oneflow {
namespace xrt {
namespace openvino {

bool OpenvinoExecutable::Run(const std::vector<Parameter>& inputs,
                             const ExecutableRunOptions& run_options,  // NOLINT
                             bool block_until_done) {
  InferenceEngine::InferRequest::Ptr infer_request = executable_network_->CreateInferRequestPtr();

  InferenceEngine::ConstInputsDataMap input_info(executable_network_->GetInputsInfo());
  InferenceEngine::ConstOutputsDataMap output_info(executable_network_->GetOutputsInfo());
  for (auto input_info_iter = input_info.begin(); input_info_iter != input_info.end();
       ++input_info_iter) {
    auto it = in_out_to_param_idx_.find(input_info_iter->first);
    CHECK(it != in_out_to_param_idx_.end());
    const int input_idx = it->second;
    CHECK_LT(input_idx, inputs.size());
    CHECK_EQ(infer_request->GetBlob(input_info_iter->first)->byteSize(),
             inputs[input_idx].byte_size());
    InferenceEngineDataDesc data_desc(inputs[input_idx].shape(), inputs[input_idx].data_type());
    InferenceEngine::TensorDesc in_desc(data_desc.precision(), data_desc.dims(),
                                        data_desc.layout());
    InferenceEngine::Blob::Ptr in_blob = ParameterToBlobPtr(inputs[input_idx], in_desc);
    infer_request->SetBlob(input_info_iter->first, in_blob);
  }

  // All return params are the results of the executable.
  this->results_ = run_options.return_params;

  for (auto output_info_iter = output_info.begin(); output_info_iter != output_info.end();
       ++output_info_iter) {
    auto it = in_out_to_param_idx_.find(output_info_iter->first);
    CHECK(it != in_out_to_param_idx_.end());
    const int output_idx = it->second;
    CHECK_LT(output_idx, this->results_.size());
    CHECK_EQ(infer_request->GetBlob(output_info_iter->first)->byteSize(),
             this->results_[output_idx].byte_size());
    InferenceEngineDataDesc data_desc(this->results_[output_idx].shape(),
                                      this->results_[output_idx].data_type());
    InferenceEngine::TensorDesc out_desc(data_desc.precision(), data_desc.dims(),
                                         data_desc.layout());
    InferenceEngine::Blob::Ptr out_blob = ParameterToBlobPtr(this->results_[output_idx], out_desc);
    infer_request->SetBlob(output_info_iter->first, out_blob);
  }

  infer_request->Infer();
  return true;
}

InferenceEngine::Blob::Ptr OpenvinoExecutable::ParameterToBlobPtr(
    const Parameter& input, const InferenceEngine::TensorDesc& in_desc) {
  const DataType& date_type = input.data_type();

  if (date_type == DataType::kChar) {
    return InferenceEngine::make_shared_blob<char>(in_desc, input.data<char>());
  } else if (date_type == DataType::kFloat) {
    return InferenceEngine::make_shared_blob<float>(in_desc, input.data<float>());
  } else if (date_type == DataType::kDouble) {
    return InferenceEngine::make_shared_blob<double>(in_desc, input.data<double>());
  } else if (date_type == DataType::kInt8) {
    return InferenceEngine::make_shared_blob<int8_t>(in_desc, input.data<int8_t>());
  } else if (date_type == DataType::kInt32) {
    return InferenceEngine::make_shared_blob<int32_t>(in_desc, input.data<int32_t>());
  } else if (date_type == DataType::kInt64) {
    return InferenceEngine::make_shared_blob<int64_t>(in_desc, input.data<int64_t>());
  } else if (date_type == DataType::kUInt8) {
    return InferenceEngine::make_shared_blob<uint8_t>(in_desc, input.data<uint8_t>());
  } else {
    UNIMPLEMENTED();
  }
  return InferenceEngine::Blob::Ptr();
}

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
