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
#define PRINT(x, len) std::cout<<#x<<" ";for(int i=0;i<len;++i) {std::cout<<x[i]<<" ";}std::cout<<std::endl;

template<typename T>
class NormalizationTrainKernel final : public user_op::OpKernel {
 public:
  NormalizationTrainKernel() = default;
  ~NormalizationTrainKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    if (ctx->op_type_name() == "normalization") { CHECK(ctx->Attr<bool>("training")); }
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x",0);
    const user_op::TensorDesc* y_desc = ctx->TensorDesc4ArgNameAndIndex("y",0);

    int32_t axis = ctx->Attr<int32_t>("axis");
    float epsilon = ctx->Attr<float>("epsilon");
    float momentum = ctx->Attr<float>("momentum");

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape(), y->shape());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());

    user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);


    // user_op::Tensor* moving_mean = nullptr;
    // user_op::Tensor* moving_variance = nullptr;
    // if (ctx->has_input("moving_mean", 0)) {
    //   CHECK(ctx->has_input("moving_variance", 0));
    //   moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
    //   moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0); 
    // }
    std::cout<<"gamma";
    PrintResult(gamma);
    std::cout<<"beta";
    PrintResult(beta);
    std::cout<<std::endl;
    NpuCommand npu_command;
    std::cout<<x->shape().ToString()<<std::endl;
    std::vector<int64_t> batch_desc = {x->shape().ptr()[1]};
    void* batch_mean_ptr = nullptr;
    uint32_t batch_mean_size = sizeof(float) * mulVector(batch_desc);
    void* batch_var_ptr = nullptr;
    uint32_t batch_var_size = sizeof(float) * mulVector(batch_desc);
    std::cout<<"batch_desc_shape "<<batch_desc[0]<<std::endl;
    AclTensorWrapper mean_warp(batch_mean_ptr,
                              ACL_FLOAT, 
                              batch_desc.size(), 
                              batch_desc.data(),
                              ACL_FORMAT_ND,
                              batch_mean_size);
    AclTensorWrapper var_warp(batch_var_ptr,
                              ACL_FLOAT, 
                              batch_desc.size(), 
                              batch_desc.data(),
                              ACL_FORMAT_ND,
                              batch_var_size);
    npu_command.OpName("BatchNorm")
               .Input(x, "channel_first")
               .Input(gamma, "channel_nd")  
               .Input(beta, "channel_nd")   
               .Output(y, "channel_first")
               .Output(mean_warp)
               .Output(var_warp)
               .Attr("epsilon", epsilon)
               .Attr("data_format", string("channels_first"))
               .Attr("is_training", true)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    PrintResult(y);
    std::cout<<"Execute Over"<<std::endl; 

//     const void* sp_alpha = CudnnSPOnePtr<T>();
//     const void* sp_beta;
//     if (ctx->has_input("_add_to_output", 0)) {
//       const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
//       CHECK_EQ(add_to_output->data_type(), y->data_type());
//       CHECK_EQ(add_to_output->shape(), y->shape());
//       Memcpy<DeviceType::kCUDA>(
//           ctx->stream(), y->mut_dptr<void>(), add_to_output->dptr<void>(),
//           add_to_output->shape().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
//       sp_beta = CudnnSPOnePtr<T>();
//     } else {
//       sp_beta = CudnnSPZeroPtr<T>();
//     }

// #if defined(BN_ENABLE_EX_API)
//     size_t workspace_size;
//     OF_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
//         ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
//         CUDNN_BATCHNORM_OPS_BN, desc_helper.xy_desc(), nullptr, desc_helper.xy_desc(),
//         desc_helper.param_desc(), nullptr, &workspace_size));
//     size_t reserve_space_size;
//     OF_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
//         ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
//         CUDNN_BATCHNORM_OPS_BN, nullptr, desc_helper.xy_desc(), &reserve_space_size));
//     auto* workspace = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
//     if (reserve_space_size == 0 && workspace_size <= workspace->shape().elem_cnt()) {
//       OF_CUDNN_CHECK(cudnnBatchNormalizationForwardTrainingEx(
//           ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
//           CUDNN_BATCHNORM_OPS_BN, sp_alpha, sp_beta, desc_helper.xy_desc(), x->dptr(), nullptr,
//           nullptr, desc_helper.xy_desc(), y->mut_dptr(), desc_helper.param_desc(), gamma->dptr(),
//           beta->dptr(), 1.0 - momentum, moving_mean ? moving_mean->mut_dptr() : NULL,
//           moving_variance ? moving_variance->mut_dptr() : NULL, epsilon, mean->mut_dptr(),
//           inv_variance->mut_dptr(), nullptr, workspace->mut_dptr(), workspace->shape().elem_cnt(),
//           nullptr, 0));
//     } else {
//       OF_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
//           ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
//           sp_alpha, sp_beta, desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), y->mut_dptr(),
//           desc_helper.param_desc(), gamma->dptr(), beta->dptr(), 1.0 - momentum,
//           moving_mean ? moving_mean->mut_dptr() : NULL,
//           moving_variance ? moving_variance->mut_dptr() : NULL, epsilon, mean->mut_dptr(),
//           inv_variance->mut_dptr()));
//     }
// #else
//     OF_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
//         ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
//         sp_alpha, sp_beta, desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), y->mut_dptr(),
//         desc_helper.param_desc(), gamma->dptr(), beta->dptr(), 1.0 - momentum,
//         moving_mean ? moving_mean->mut_dptr() : NULL,
//         moving_variance ? moving_variance->mut_dptr() : NULL, epsilon, mean->mut_dptr(),
//         inv_variance->mut_dptr()));
// #endif

//     if (ctx->op_type_name() == "normalization_add_relu") {
//       CHECK(!ctx->has_input("_add_to_output", 0));
//       const int64_t elem_cnt = x->shape().elem_cnt();
//       auto* mask = ctx->Tensor4ArgNameAndIndex("reserve_space", 0);
//       if (ctx->has_input("addend", 0)) {
//         const auto* addend = ctx->Tensor4ArgNameAndIndex("addend", 0);
//         AddRelu(ctx->stream(), elem_cnt, y->dptr<T>(), addend->dptr<T>(), y->mut_dptr<T>(),
//                 mask->mut_dptr<int32_t>());
//       } else {
//         Relu(ctx->stream(), elem_cnt, y->dptr<T>(), y->mut_dptr<T>(), mask->mut_dptr<int32_t>());
//       }
//     }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_TRAIN_KERNEL(dtype)                                                         \
  REGISTER_USER_KERNEL("normalization")                                                         \
      .SetCreateFn<NormalizationTrainKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                          \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)           \
                       && (user_op::HobAttr<bool>("training") == true));                         \
    //   .SetInferTmpSizeFn(InferTrainTmpSize)                                                     \
    //   .SetInplaceProposalFn([](const user_op::InferContext& ctx,                                \
    //                            user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
    //     if (ctx.has_input("_add_to_output", 0)) {                                               \
    //       OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true));           \
    //     }                                                                                       \
    //     return Maybe<void>::Ok();                                                               \
    //   });

REGISTER_BN_TRAIN_KERNEL(float16)
REGISTER_BN_TRAIN_KERNEL(float)
REGISTER_BN_TRAIN_KERNEL(double)

}// {anonymous} namespace
}// namespace oneflow

#endif// WITH_NPU