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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/user/kernels/loss_kernel_util.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {
namespace user_op {
namespace {

using namespace loss;


template<typename T, typename K>
class NllKernel final : public user_op::OpKernel {
 public:
  NllKernel() = default;
  ~NllKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor*  input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor*  target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* total_weight_blob = ctx->Tensor4ArgNameAndIndex("total_weight", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    int64_t ignore_index = ctx->Attr<int64_t>("ignore_index");
    std::string reduction = ctx->Attr<std::string>("reduction");
    user_op::Tensor* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0) : nullptr;

    if(!weight)
    {
        std::vector<float> weight_v(input_blob->shape_view().elem_cnt(), 1.0);
        std::vector<int64_t> weight_shape;
        weight_shape.push_back(input_blob->shape_view().At(1));
        CHECK_EQ(input_blob->shape_view().At(1)*sizeof(float), tmp_buffer->shape_view().elem_cnt() );
        OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));
        AclTensorWrapper wrap(tmp_buffer->mut_dptr<void>(), ACL_FLOAT, weight_shape.size(), weight_shape.data(), ACL_FORMAT_ND,
                                input_blob->shape_view().At(1)*sizeof(float), weight_v.data());
        OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));                                
        NpuCommand npu_command;
        npu_command.OpName("NLLLoss")
                  .Input(input_blob)
                  .Input(target_blob)
                  .Input(wrap)
                  .Attr("reduction",reduction)
                  .Attr("ignore_index",ignore_index)
                  .Output(out_blob)
                  .Output(total_weight_blob)
                  .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                  .Check();
        npu_command.Run()
               .Realease();
      //   OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));  
      //   PrintResult(out_blob);
      //  std::cout<<"NllKernel Execute Over"<<std::endl;  
    }
    else
    {
        UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T, typename K>
class NllGradKernel final : public user_op::OpKernel {
 public:
  NllGradKernel() = default;
  ~NllGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("out_grad", 0);
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("in_grad", 0);
    user_op::Tensor* total_weight_blob = ctx->Tensor4ArgNameAndIndex("total_weight", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    int64_t ignore_index = ctx->Attr<int64_t>("ignore_index");
    std::string reduction = "mean";//ctx->Attr<std::string>("reduction");
    user_op::Tensor* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0) : nullptr;
    if(!weight)
    {
        std::vector<float> weight_v(input_blob->shape_view().At(1), 1.0);
        std::vector<int64_t> weight_shape;
        weight_shape.push_back(input_blob->shape_view().At(1));
        CHECK_EQ(tmp_buffer->shape_view().elem_cnt(), input_blob->shape_view().At(1)*sizeof(float));
        // dck_caution_here : use Fill to replace aclrtMemcpy
        OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));
        AclTensorWrapper wrap(tmp_buffer->mut_dptr<void>(), ACL_FLOAT, weight_shape.size(), weight_shape.data(), ACL_FORMAT_ND,
                                input_blob->shape_view().At(1)*sizeof(float), weight_v.data());
        OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));
        NpuCommand npu_command;
        npu_command.OpName("NLLLossGrad")
                .Input(input_blob)
                .Input(dy_blob)
                .Input(target_blob)
                .Input(wrap)
                .Input(total_weight_blob)
                .Attr("reduction",reduction)
                .Attr("ignore_index",ignore_index)
                .Output(dx_blob)
                .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                .Check();
        npu_command.Run()
               .Realease();
        // OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));  
        // PrintResult(dx_blob);
        // std::cout<<"NLLLossGrad over"<<std::endl;
    }
    else
    {
      UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace
#define REGISTER_NLL_KERNEL(dtype_pair, ltype_pair)                                            \
  REGISTER_USER_KERNEL("nll_npu")                                                                  \
      .SetCreateFn<NllKernel<OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(ltype_pair)>>()    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                          \
                       && (user_op::HobDataType("target", 0) == OF_PP_PAIR_SECOND(ltype_pair)) \
                       && (user_op::HobDataType("output", 0) == OF_PP_PAIR_SECOND(dtype_pair)))    \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t{                              \
          const auto& x = ctx->InputTensorDesc("input", 0);                                   \
          size_t tmp_size = 0;                                                                  \
          int shape_size = x.shape().At(1) * sizeof(float);                                     \
          tmp_size += shape_size;                                                               \
          return tmp_size;                                                                      \
      });   
#define REGISTER_NLL_GRAD_KERNEL(dtype_pair, ltype_pair)                                        \
  REGISTER_USER_KERNEL("nll_grad_npu")                                                              \
      .SetCreateFn<NllGradKernel<OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(ltype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                           \
                       && (user_op::HobDataType("target", 0) == OF_PP_PAIR_SECOND(ltype_pair))  \
                       && (user_op::HobDataType("out_grad", 0) == OF_PP_PAIR_SECOND(dtype_pair))\
                       && (user_op::HobDataType("in_grad", 0) == OF_PP_PAIR_SECOND(dtype_pair)))\
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t{                              \
          const auto& x = ctx->InputTensorDesc("input", 0);                                     \
          size_t tmp_size = 0;                                                                  \
          int shape_size = x.shape().At(1) * sizeof(float);                                     \
          tmp_size += shape_size;                                                               \
          return tmp_size;                                                                      \
      });   
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_NLL_KERNEL, FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_NLL_KERNEL, FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_NLL_GRAD_KERNEL, FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_NLL_GRAD_KERNEL, FLOAT16_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
}  // namespace user_op
}  // namespace oneflow
