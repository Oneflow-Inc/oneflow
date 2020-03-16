#include "oneflow/xrt/openvino/openvino_executable.h"
#include "oneflow/xrt/platform.h"

namespace oneflow {
namespace xrt {
namespace openvino {

bool OpenvinoExecutable::Run(const std::vector<Parameter> &inputs,
                             const ExecutableRunOptions &run_options,  // NOLINT
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
    InferenceEngine::Blob::Ptr in_blob =
        InferenceEngine::make_shared_blob<float>(in_desc, inputs[input_idx].data<float>());
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
    InferenceEngine::Blob::Ptr out_blob = InferenceEngine::make_shared_blob<float>(
        out_desc, this->results_[output_idx].data<float>());
    infer_request->SetBlob(output_info_iter->first, out_blob);
  }

  infer_request->Infer();
  return true;
}

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
