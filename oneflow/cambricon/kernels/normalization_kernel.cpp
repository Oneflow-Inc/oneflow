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
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/core/ep/include/primitive/permute.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(Context* ctx, const int& num_dims) {
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(ctx->device_type(), num_dims);
}

void infer_channel_sizes(const ShapeView& shape, int* n, int* c, int* h, int* w) {
  if (shape.NumAxes() == 2) {
    *n = shape.At(0);
    *h = 1;
    *w = 1;
    *c = shape.At(1);
  } else {
    *n = shape.At(0);
    *c = shape.At(1);
    *h = shape.At(2);
    *w = shape.At(3);
  }
}

}  // namespace

template<typename T>
class MluNormalizationInferenceKernel final : public user_op::OpKernel {
 public:
  MluNormalizationInferenceKernel() = default;
  ~MluNormalizationInferenceKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const bool training = ctx->Attr<bool>("training");
    CHECK(!training);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
    auto* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    // make sure input tensor's format NCHW, so channel axis must be 1
    const auto axis = ctx->Attr<int32_t>("axis");
    CHECK_EQ(axis, 1);
    const auto epsilon = ctx->Attr<float>("epsilon");

    int n = 0, c = 0, h = 0, w = 0;
    infer_channel_sizes(x->shape_view(), &n, &c, &h, &w);

    size_t tmp_in_size = x->shape_view().elem_cnt() * GetSizeOfDataType(x->data_type());
    size_t tmp_out_size = y->shape_view().elem_cnt() * GetSizeOfDataType(y->data_type());
    CnnlWorkspace tmp_in_workspace(ctx->stream()->As<ep::MluStream>(), tmp_in_size);
    CnnlWorkspace tmp_out_workspace(ctx->stream()->As<ep::MluStream>(), tmp_out_size);
    void* tmp_in_dptr = tmp_in_workspace.dptr();
    void* tmp_out_dptr = tmp_out_workspace.dptr();

    auto transpose = NewPermutePrimitive(ctx, x->shape_view().NumAxes());
    CHECK(transpose);

    int permute_nhwc[4] = {0, 2, 3, 1};
    // transpose input NCHW -> NHWC
    transpose->Launch(ctx->stream(), x->data_type(), x->shape_view().NumAxes(),
                      x->shape_view().data(), x->dptr<T>(), permute_nhwc, tmp_in_dptr);

    int64_t shape_nhwc[4] = {n, h, w, c};
    CnnlTensorDescriptor input_desc, output_desc, weight_bias_mean_var_desc;
    auto dtype = ConvertToCnnlDataType(x->data_type());
    input_desc.set(4, shape_nhwc, dtype, CNNL_LAYOUT_NHWC);
    output_desc.set(4, shape_nhwc, dtype, CNNL_LAYOUT_NHWC);
    int dim[1] = {c};
    weight_bias_mean_var_desc.set(1, dim, dtype, CNNL_LAYOUT_ARRAY);
    // api reference:
    // https://www.cambricon.com/docs/sdk_1.10.0/cambricon_cnnl_1.15.2/developer_guide/cnnl_api/api/batchnorm.html#cnnlbatchnormforwardinference
    // inference
    OF_CNNL_CHECK(cnnlBatchNormForwardInference(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), nullptr, nullptr, input_desc.desc(),
        tmp_in_dptr, weight_bias_mean_var_desc.desc(), gamma->dptr(), beta->dptr(),
        moving_mean->dptr(), moving_variance->dptr(), epsilon, output_desc.desc(), tmp_out_dptr));

    int permute_nchw[4] = {0, 3, 1, 2};
    // transpose output NHWC -> NCHW
    transpose->Launch(ctx->stream(), y->data_type(), y->shape_view().NumAxes(), shape_nhwc,
                      tmp_out_dptr, permute_nchw, y->mut_dptr());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_INFERENCE_MLU_KERNEL(dtype)                                       \
  REGISTER_USER_KERNEL("normalization")                                               \
      .SetCreateFn<MluNormalizationInferenceKernel<dtype>>()                          \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                 \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobAttr<bool>("training") == false));

REGISTER_BN_INFERENCE_MLU_KERNEL(float)
REGISTER_BN_INFERENCE_MLU_KERNEL(float16)

#undef REGISTER_BN_INFERENCE_MLU_KERNEL

template<typename T>
class MluNormalizationTrainingKernel final : public user_op::OpKernel {
 public:
  MluNormalizationTrainingKernel() = default;
  ~MluNormalizationTrainingKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const bool training = ctx->Attr<bool>("training");
    CHECK(training);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const auto momentum = ctx->Attr<float>("momentum");

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape_view(), y->shape_view());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_LT(axis, x->shape_view().NumAxes());
    // make sure input tensor's format NCHW, so channel axis must be 1
    CHECK_EQ(axis, 1);

    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);

    user_op::Tensor* moving_mean = nullptr;
    user_op::Tensor* moving_variance = nullptr;
    if (ctx->has_input("moving_mean", 0)) {
      CHECK(ctx->has_input("moving_variance", 0));
      moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
      moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    }
    void* moving_mean_ptr = nullptr;
    void* moving_variance_ptr = nullptr;
    if (moving_mean != nullptr && moving_variance != nullptr) {
      moving_mean_ptr = (void*)moving_mean->mut_dptr();
      moving_variance_ptr = (void*)moving_variance->mut_dptr();
    }

    int n = 0, c = 0, h = 0, w = 0;
    infer_channel_sizes(x->shape_view(), &n, &c, &h, &w);

    size_t tmp_in_size = x->shape_view().elem_cnt() * GetSizeOfDataType(x->data_type());
    size_t tmp_out_size = y->shape_view().elem_cnt() * GetSizeOfDataType(y->data_type());
    CnnlWorkspace tmp_in_workspace(ctx->stream()->As<ep::MluStream>(), tmp_in_size);
    CnnlWorkspace tmp_out_workspace(ctx->stream()->As<ep::MluStream>(), tmp_out_size);
    void* tmp_in_dptr = tmp_in_workspace.dptr();
    void* tmp_out_dptr = tmp_out_workspace.dptr();

    auto transpose = NewPermutePrimitive(ctx, x->shape_view().NumAxes());
    CHECK(transpose);
    int permute_nhwc[4] = {0, 2, 3, 1};
    // transpose input NCHW -> NHWC
    transpose->Launch(ctx->stream(), x->data_type(), x->shape_view().NumAxes(),
                      x->shape_view().data(), x->dptr<T>(), permute_nhwc, tmp_in_dptr);

    int64_t shape_nhwc[4] = {n, h, w, c};
    CnnlTensorDescriptor input_desc, output_desc, weight_bias_mean_var_desc;
    auto dtype = ConvertToCnnlDataType(x->data_type());
    input_desc.set(4, shape_nhwc, dtype, CNNL_LAYOUT_NHWC);
    output_desc.set(4, shape_nhwc, dtype, CNNL_LAYOUT_NHWC);
    int dim[1] = {c};
    weight_bias_mean_var_desc.set(1, dim, dtype, CNNL_LAYOUT_ARRAY);

    // api reference:
    // https://www.cambricon.com/docs/sdk_1.10.0/cambricon_cnnl_1.15.2/developer_guide/cnnl_api/api/batchnorm.html#cnnlbatchnormforwardtraining
    // training
    OF_CNNL_CHECK(cnnlBatchNormForwardTraining(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), nullptr, nullptr, input_desc.desc(),
        tmp_in_dptr, weight_bias_mean_var_desc.desc(), gamma->dptr(), beta->dptr(), moving_mean_ptr,
        moving_variance_ptr, epsilon, momentum, output_desc.desc(), tmp_out_dptr, mean->mut_dptr(),
        inv_variance->mut_dptr()));

    // transpose output NHWC -> NCHW
    int permute_nchw[4] = {0, 3, 1, 2};
    transpose->Launch(ctx->stream(), y->data_type(), y->shape_view().NumAxes(), shape_nhwc,
                      tmp_out_dptr, permute_nchw, y->mut_dptr());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_TRAINING_MLU_KERNEL(dtype)                                        \
  REGISTER_USER_KERNEL("normalization")                                               \
      .SetCreateFn<MluNormalizationTrainingKernel<dtype>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                 \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobAttr<bool>("training") == true));

REGISTER_BN_TRAINING_MLU_KERNEL(float)
REGISTER_BN_TRAINING_MLU_KERNEL(float16)

#undef REGISTER_BN_TRAINING_MLU_KERNEL

}  // namespace oneflow
