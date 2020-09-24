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
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class TensorListToTensorBufferKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorListToTensorBufferKernel);
  TensorListToTensorBufferKernel() = default;
  ~TensorListToTensorBufferKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardHeader(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

void TensorListToTensorBufferKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CHECK_EQ(in_blob->total_num_of_tensors(), out_blob->shape().elem_cnt())
      << "out_blob shape: " << out_blob->shape().ToString();

  TensorView in_tensor_view = in_blob->BeginTensor();
  TensorBuffer* out_buffer = out_blob->mut_dptr<TensorBuffer>();
  while (!in_blob->IsEndTensor(in_tensor_view)) {
    CHECK_EQ(in_tensor_view.shape().At(0), 1);
    DimVector dim_vec(in_tensor_view.shape().NumAxes() - 1);
    FOR_RANGE(int, i, 0, dim_vec.size()) { dim_vec.at(i) = in_tensor_view.shape().At(i + 1); }
    out_buffer->Resize(Shape(dim_vec), in_blob->data_type());
    memcpy(out_buffer->mut_data(), in_tensor_view.dptr(), out_buffer->nbytes());
    out_buffer += 1;
    in_blob->MoveToNextTensor(&in_tensor_view);
  }
}

void TensorListToTensorBufferKernel::ForwardHeader(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(!this->kernel_conf().need_do_opaque_header());
  BnInOp2Blob("out")->mut_shape_view()->Set(0, BnInOp2Blob("in")->total_num_of_tensors());
}

NEW_REGISTER_KERNEL(OperatorConf::kTensorListToTensorBufferConf, TensorListToTensorBufferKernel)
    .SetIsMatchedPred([](const KernelConf& conf) {
      return (conf.op_attribute().op_conf().device_tag() == "cpu")
             && (conf.data_type() == DataType::kTensorBuffer);
    });

}  // namespace oneflow
