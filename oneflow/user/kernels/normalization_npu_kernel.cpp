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
#ifdef WITH_NPU

#include "oneflow/core/framework/framework.h"
#include "oneflow/user/ops/npu_command.h"

namespace oneflow {

namespace {
template<typename T>
class NormalizationInferenceNpuKernel final : public user_op::OpKernel {
 public:
  NormalizationInferenceNpuKernel() = default;
  ~NormalizationInferenceNpuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const bool training = ctx->Attr<bool>("training");
    CHECK(!training);
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    user_op::Tensor* moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
    user_op::Tensor* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape_view(), y->shape_view());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape_view().NumAxes());

    if (axis == 1) {  // NOTE(Liang Depeng): NCHW format
        NpuCommand npu_command;
        npu_command.OpName("BNInfer")
                   .Input(x,"channels_fisrt")
                   .Input(gamma,"channels_fisrt")
                   .Input(beta,"channels_fisrt")
                   .Input(moving_mean)
                   .Input(moving_variance)
                   .Output(y)
                   .Attr("epsilon",epsilon)
                   .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                   .Check();
        npu_command.Run()
               .Realease();
        //OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    } else {  // TODO(Liang Depeng): NHWC format
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_INFERENCE_CPU_KERNEL(dtype)                                           \
  REGISTER_USER_KERNEL("normalization")                                                   \
      .SetCreateFn<NormalizationInferenceNpuKernel<dtype>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                     \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobAttr<bool>("training") == false));
      // .SetInplaceProposalFn(                                                              \
      //     [](const user_op::InferContext& ctx,                                            \
      //        const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {      \
      //       if (ctx.has_input("_add_to_output", 0)) {                                     \
      //         OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true)); \
      //       }                                                                             \
      //       return Maybe<void>::Ok();                                                     \
      //     });
REGISTER_BN_INFERENCE_CPU_KERNEL(float16)
REGISTER_BN_INFERENCE_CPU_KERNEL(float)
REGISTER_BN_INFERENCE_CPU_KERNEL(double)

#undef REGISTER_BN_INFERENCE_CPU_KERNEL

template<typename T>
class NormalizationTrainKernel final : public user_op::OpKernel {
 public:
  NormalizationTrainKernel() = default;
  ~NormalizationTrainKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override ;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
template<typename T>
void NormalizationTrainKernel<T>::Compute(user_op::KernelComputeContext* ctx) const
{
     if (ctx->op_type_name() == "normalization") { CHECK(ctx->Attr<bool>("training")); }
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    int32_t axis = ctx->Attr<int32_t>("axis");
    float epsilon = ctx->Attr<float>("epsilon");
    float momentum = ctx->Attr<float>("momentum");

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape_view(), y->shape_view());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape_view().NumAxes());

    user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    
    std::vector<int64_t> batch_desc = {x->shape_view().ptr()[axis]};
    size_t len_of_wrap = sizeof(float) * mulVector(batch_desc);
    char* tmp_ptr = tmp_buffer->mut_dptr<char>();
    AclTensorWrapper sum_warp(static_cast<void*>(tmp_ptr),
                              ACL_FLOAT, 
                              batch_desc.size(), 
                              batch_desc.data(),
                              ACL_FORMAT_NCHW,
                              len_of_wrap);
    AclTensorWrapper square_sum_warp(static_cast<void*>(tmp_ptr+len_of_wrap),
                              ACL_FLOAT, 
                              batch_desc.size(), 
                              batch_desc.data(),
                              ACL_FORMAT_NCHW,
                              len_of_wrap);
    NpuCommand reduce_command;
    reduce_command.OpName("BNTrainingReduce")
               .Input(x,"channels_first")
               .Output(sum_warp)
               .Output(square_sum_warp)
               .Attr("epsilon", epsilon)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    reduce_command.Run()
               .Realease();
    user_op::Tensor* moving_mean = nullptr;
    user_op::Tensor* moving_variance = nullptr;
    if (ctx->has_input("moving_mean", 0)) {
      CHECK(ctx->has_input("moving_variance", 0));
      moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
      moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0); 
    }
    
    NpuCommand update_command;
    update_command.OpName("BNTrainingUpdate")
                  .Input(x,"channels_first")
                  .Input(sum_warp)
                  .Input(square_sum_warp)
                  .Input(gamma,"channels_first")
                  .Input(beta, "channels_first")
                  .Input(moving_mean,"channels_first")
                  .Input(moving_variance,"channels_first")
                  .Output(y,"channels_first")
                  .Output(moving_mean,"channels_first")
                  .Output(moving_variance,"channels_first")
                  .Output(mean,"channels_first")
                  .Output(inv_variance,"channels_first")
                  .Attr("epsilon",epsilon)
                  .Attr("factor",momentum)
                  .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                  .Check();
    update_command.Run()
               .Realease();
}
#define REGISTER_BN_TRAIN_KERNEL(dtype)                                                         \
  REGISTER_USER_KERNEL("normalization")                                                         \
      .SetCreateFn<NormalizationTrainKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                           \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)           \
                       && (user_op::HobAttr<bool>("training") == true))                         \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t{                              \
          const auto& x = ctx->InputTensorDesc("x", 0);                                         \
          size_t tmp_size = 0;                                                                  \
          int channel_size = x.shape().At(1) * sizeof(float); /* only support channels first*/  \
          tmp_size += 2 * channel_size; /* for sum and square_sum */                            \
          return tmp_size;                                                                      \
      });                                                    

REGISTER_BN_TRAIN_KERNEL(float16)
REGISTER_BN_TRAIN_KERNEL(float)
REGISTER_BN_TRAIN_KERNEL(double)


template<typename T>
class NormalizationGradNpuKernel final : public user_op::OpKernel {
 public:
  NormalizationGradNpuKernel() = default;
  ~NormalizationGradNpuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    user_op::Tensor* gamma_diff = ctx->Tensor4ArgNameAndIndex("gamma_diff", 0);
    user_op::Tensor* beta_diff = ctx->Tensor4ArgNameAndIndex("beta_diff", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    //auto* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto axis = ctx->Attr<int32_t>("axis");
    float epsilon = ctx->Attr<float>("epsilon");

    const DataType data_type = x->data_type();
    CHECK_EQ(dy->shape_view(), x->shape_view());
    CHECK_EQ(dy->data_type(), data_type);
    CHECK_EQ(dx->shape_view(), x->shape_view());
    CHECK_EQ(dx->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape_view().NumAxes());

    if (ctx->op_type_name() == "normalization_grad") {
      // do nothing for npu
    } else if (ctx->op_type_name() == "normalization_add_relu_grad") {
      UNIMPLEMENTED();
    } else {
      UNIMPLEMENTED();
    }
    if (axis == 1) {  // NOTE(Liang Depeng): NCHW format
      NpuCommand update_command;
      update_command.OpName("BNTrainingUpdateGrad")
                    .Input(dy,"channels_first")
                    .Input(x,"channels_first")
                    .Input(mean,"channels_first")
                    .Input(inv_variance,"channels_first")
                    .Output(gamma_diff,"channels_first")
                    .Output(beta_diff,"channels_first")
                    .Attr("epsilon",epsilon)
                    .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                    .Check();
      update_command.Run()
               .Realease();
      NpuCommand reduce_command;
      reduce_command.OpName("BNTrainingReduceGrad")
                    .Input(dy,"channels_first")
                    .Input(x,"channels_first")
                    .Input(gamma_diff,"channels_first")
                    .Input(beta_diff,"channels_first")
                    .Input(gamma,"channels_first")
                    .Input(mean,"channels_first")
                    .Input(inv_variance,"channels_first")
                    .Output(dx,"channels_first")
                    .Attr("epsilon",epsilon)
                    .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                    .Check();
      reduce_command.Run()
               .Realease();        
    } else {  // TODO(Liang Depeng): NHWC format
      UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_GRAD_NPU_KERNEL(dtype)                            \
  REGISTER_USER_KERNEL("normalization_grad")                          \
      .SetCreateFn<NormalizationGradNpuKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));
REGISTER_BN_GRAD_NPU_KERNEL(float16)
REGISTER_BN_GRAD_NPU_KERNEL(float)
REGISTER_BN_GRAD_NPU_KERNEL(double)

}// {anonymous} namespace
}// namespace oneflow

#endif// WITH_NPU