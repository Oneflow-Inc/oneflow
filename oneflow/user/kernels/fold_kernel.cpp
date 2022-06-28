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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/user/kernels/fold_kernel_util.h"

namespace oneflow {

namespace user_op {

namespace {

// NDIM range: (1, 2, 3)
// SDIM range: (1, 2), 1 indicates channels_last, 2 indicates channels_first
template<typename INDEX_T, int NDIM, int SDIM>
class FoldOpKernelState : public OpKernelState {
 public:
  using ParamType = FoldParams<INDEX_T, NDIM, SDIM>;
  FoldOpKernelState(const ShapeView& input_shape, const std::vector<int32_t>& output_size,
                    const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& padding,
                    const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation)
      : params_(input_shape.At(0), input_shape.At(ParamType::kInputChannelDim), output_size.data(),
                input_shape.ptr() + SDIM, kernel_size.data(), padding.data(), stride.data(),
                dilation.data()) {}
  const ParamType& params() const { return params_; }

 private:
  ParamType params_;
};

template<typename INDEX_T, int NDIM, int SDIM>
std::shared_ptr<FoldOpKernelState<INDEX_T, NDIM, SDIM>> CreateFoldOpKernelState(
    const ShapeView& input_shape, const std::vector<int32_t>& output_size,
    const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& padding,
    const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation) {
  std::shared_ptr<FoldOpKernelState<INDEX_T, NDIM, SDIM>> state(
      new FoldOpKernelState<INDEX_T, NDIM, SDIM>(input_shape, output_size, kernel_size, padding,
                                                 stride, dilation));
  return state;
}

template<DeviceType device_type, typename T, typename INDEX_T, int NDIM, int SDIM>
class FoldKernel final : public OpKernel {
 public:
  FoldKernel() = default;
  ~FoldKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* input = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* output = ctx->Tensor4ArgNameAndIndex("y", 0);

    const std::vector<int32_t> output_size = ctx->Attr<std::vector<int32_t>>("output_size");
    const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
    const std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("strides");

    const auto& state_ptr = CreateFoldOpKernelState<INDEX_T, NDIM, SDIM>(
        input->shape_view(), output_size, kernel_size, padding, stride, dilation);
    const FoldParams<INDEX_T, NDIM, SDIM> params = state_ptr->params();
    size_t out_bytes_size =
        output->shape_view().elem_cnt() * GetSizeOfDataType(output->data_type());
    Memset<device_type>(ctx->stream(), output->mut_dptr<T>(), 0, out_bytes_size);
    FoldKernelUtil<device_type, T, INDEX_T, NDIM, SDIM>::Forward(
        ctx->stream(), &params, input->dptr<T>(), output->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

// Currently support 4-D tensor and NCHW format
#define REGISTER_FOLD_KERNEL(device, dtype)                    \
  REGISTER_USER_KERNEL("fold")                                 \
      .SetCreateFn<FoldKernel<device, dtype, int32_t, 2, 2>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)    \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_FOLD_KERNEL(DeviceType::kCPU, float)
REGISTER_FOLD_KERNEL(DeviceType::kCPU, double)

#ifdef WITH_CUDA
REGISTER_FOLD_KERNEL(DeviceType::kCUDA, float)
REGISTER_FOLD_KERNEL(DeviceType::kCUDA, double)
#endif  // WITH_CUDA

}  // namespace user_op

}  // namespace oneflow
