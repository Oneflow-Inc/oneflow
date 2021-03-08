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
#ifdef WITH_CUDA

#include "oneflow/core/framework/framework.h"
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/device/cudnn_conv_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename PerfT>
struct CudnnConvArgsAndAlgo final {
  using AlgoT = decltype(std::declval<PerfT>().algo);

  CudnnConvArgs args;
  PerfT algo_perf;

  CudnnConvArgsAndAlgo(const user_op::Tensor* x, const user_op::Tensor* w, const user_op::Tensor* y,
                       user_op::Tensor* buf, const JobDesc& job_desc,
                       const user_op::UserOpConfWrapper& user_op_conf, DeviceCtx* device_ctx,
                       bool has_forced_algo, int32_t forced_algo)
      : args(user_op_conf, x->data_type(), x->shape(), w->data_type(), w->shape(), y->data_type(),
             y->shape(), user_op_conf.attr<std::string>("data_format"), buf->shape().elem_cnt(),
             job_desc.job_conf().cudnn_conv_heuristic_search_algo(),
             job_desc.job_conf().cudnn_conv_use_deterministic_algo_only(),
             job_desc.job_conf().cudnn_conv_enable_pseudo_half()
                 || (user_op_conf.attr<std::string>("data_format") == "channels_last"
                     && std::is_same<PerfT, cudnnConvolutionBwdFilterAlgoPerf_t>::value)) {
    size_t byte_size_of_buf = buf->shape().elem_cnt();
    AllocatedCudnnConvResource res(device_ctx->cudnn_handle(), const_cast<void*>(x->dptr()),
                                   const_cast<void*>(w->dptr()), const_cast<void*>(y->dptr()),
                                   buf->mut_dptr());
    if (has_forced_algo) {
      algo_perf = GetCudnnConvAlgorithmPerferenceWithResource<PerfT>(
          &args, &res, static_cast<AlgoT>(forced_algo));
    } else {
      algo_perf = FindCudnnConvAlgorithmWithResource<PerfT>(&args, &res);
    }
    CHECK_EQ(algo_perf.status, CUDNN_STATUS_SUCCESS)
        << "op (" << user_op_conf.op_name()
        << ") find algorithm perference failed. algo: " << algo_perf.algo;
    CHECK_LE(algo_perf.memory, byte_size_of_buf)
        << "op (" << user_op_conf.op_name() << ") find algorithm " << algo_perf.algo
        << ", need memory " << algo_perf.memory << ", but cudnn_buf_limit_byte is "
        << byte_size_of_buf;
    OF_CUDNN_CHECK(cudnnSetConvolutionMathType(args.cdesc.Get(), algo_perf.mathType));
  }
  CudnnConvArgsAndAlgo() = delete;
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvArgsAndAlgo);
};

template<typename PerfT>
size_t InferTmpSizeWithCudnn(const user_op::TensorDesc* x, const user_op::TensorDesc* w,
                             const user_op::TensorDesc* y, const JobDesc& job_desc,
                             const user_op::UserOpConfWrapper& user_op_conf, bool has_forced_algo,
                             int32_t forced_algo) {
  using AlgoT = decltype(std::declval<PerfT>().algo);

  size_t workspace_size = job_desc.cudnn_buf_limit_mbyte() * 1024 * 1024;
  if (!x->is_dynamic()) {
    CudnnConvArgs args(user_op_conf, x->data_type(), ShapeView(x->shape()), w->data_type(),
                       ShapeView(w->shape()), y->data_type(), ShapeView(y->shape()),
                       user_op_conf.attr<std::string>("data_format"), workspace_size,
                       job_desc.job_conf().cudnn_conv_heuristic_search_algo(),
                       job_desc.job_conf().cudnn_conv_use_deterministic_algo_only(),
                       job_desc.job_conf().cudnn_conv_enable_pseudo_half()
                           || (user_op_conf.attr<std::string>("data_format") == "channels_last"
                               && std::is_same<PerfT, cudnnConvolutionBwdFilterAlgoPerf_t>::value));
    PerfT algo_perf;
    if (has_forced_algo) {
      algo_perf = GetCudnnConvAlgorithmPerference<PerfT>(&args, static_cast<AlgoT>(forced_algo));
    } else {
      algo_perf = FindCudnnConvAlgorithm<PerfT>(&args);
    }
    CHECK_EQ(algo_perf.status, CUDNN_STATUS_SUCCESS)
        << "op (" << user_op_conf.op_name()
        << ") find algorithm perference failed. algo: " << algo_perf.algo;
    CHECK_LE(algo_perf.memory, workspace_size)
        << "op (" << user_op_conf.op_name() << ") find algorithm " << algo_perf.algo
        << ", need memory " << algo_perf.memory << ", but cudnn_buf_limit_byte is "
        << workspace_size;
    workspace_size = algo_perf.memory;
  }
  workspace_size = std::max(size_t(1), workspace_size);
  return workspace_size;
}

// for 1d and 2d
template<size_t NDims>
CudnnTensorDesc* GetBiasCudnnTensorDesc(const std::string& data_format, int32_t filters,
                                        DataType data_type) {
  if (data_format == "channels_first") {
    return new CudnnTensorDesc(CUDNN_TENSOR_NCHW, data_type, 1, filters, 1, 1);
  } else {
    CHECK_EQ("channels_last", data_format);
    CHECK_EQ(DataType::kFloat, data_type)
        << "CUDNN 1d & 2d support channels last only if data type is float";
    return new CudnnTensorDesc(CUDNN_TENSOR_NHWC, data_type, 1, filters, 1, 1);
  }
}

// for 3d and Nd
template<>
CudnnTensorDesc* GetBiasCudnnTensorDesc<3>(const std::string& data_format, int32_t filters,
                                           DataType data_type) {
  constexpr int NDims = 3 + 2;
  CHECK_EQ("channels_first", data_format) << "CUDNN Nd API only support channels first";
  std::vector<int32_t> bias_dim(NDims, 1);
  std::vector<int32_t> stride_of_bias_tensor(NDims, 1);
  bias_dim[1] = filters;
  stride_of_bias_tensor[0] = filters;
  return new CudnnTensorDesc(data_type, NDims, bias_dim.data(), stride_of_bias_tensor.data());
}

struct ConvCudnnOpKernelState final : public user_op::OpKernelState {
  std::unique_ptr<CudnnTensorDesc> bias_desc;
};

template<typename T, size_t NDims>
class ConvGpuKernel final : public user_op::OpKernel {
 public:
  ConvGpuKernel() = default;
  ~ConvGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const {
    const auto& data_format = ctx->Attr<std::string>("data_format");
    int32_t filters = ctx->Attr<int32_t>("filters");

    std::shared_ptr<ConvCudnnOpKernelState> state(new ConvCudnnOpKernelState());

    const user_op::TensorDesc* bias = ctx->TensorDesc4ArgNameAndIndex("bias", 0);
    if (bias != nullptr) {
      state->bias_desc.reset(
          GetBiasCudnnTensorDesc<NDims>(data_format, filters, GetDataType<T>::value));
    }

    return std::move(state);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const JobDesc& job_desc = ctx->job_desc();

    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CudnnConvArgsAndAlgo<cudnnConvolutionFwdAlgoPerf_t> args_and_algo(
        in, weight, out, buf, job_desc, ctx->user_op_conf(), ctx->device_ctx(),
        job_desc.job_conf().has_cudnn_conv_force_fwd_algo(),
        job_desc.job_conf().cudnn_conv_force_fwd_algo());
    const CudnnConvArgs& args = args_and_algo.args;
    const cudnnConvolutionFwdAlgoPerf_t& algo_perf = args_and_algo.algo_perf;

    OF_CUDNN_CHECK(cudnnConvolutionForward(
        ctx->device_ctx()->cudnn_handle(), CudnnSPOnePtr<T>(), args.xdesc.Get(), in->dptr(),
        args.wdesc.Get(), weight->dptr(), args.cdesc.Get(), algo_perf.algo, buf->mut_dptr(),
        args.params.max_ws_size, CudnnSPZeroPtr<T>(), args.ydesc.Get(), out->mut_dptr()));

    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    if (bias != nullptr) {
      ConvCudnnOpKernelState* conv_state = dynamic_cast<ConvCudnnOpKernelState*>(state);
      CHECK_NOTNULL(conv_state);
      OF_CUDNN_CHECK(cudnnAddTensor(ctx->device_ctx()->cudnn_handle(), CudnnSPOnePtr<T>(),
                                    conv_state->bias_desc->Get(), bias->dptr<T>(),
                                    CudnnSPOnePtr<T>(), args.ydesc.Get(), out->mut_dptr<T>()));
    }
  }
};

#define REGISTER_CONV_KERNEL(op_name, dtype, ndims)                                    \
  REGISTER_USER_KERNEL(#op_name)                                                       \
      .SetCreateFn<ConvGpuKernel<dtype, ndims>>()                                      \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                    \
        const JobDesc& job_desc = *ctx->job_desc();                                    \
        const auto* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);                     \
        const auto* weight = ctx->TensorDesc4ArgNameAndIndex("weight", 0);             \
        const auto* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);                   \
        return InferTmpSizeWithCudnn<cudnnConvolutionFwdAlgoPerf_t>(                   \
            in, weight, out, job_desc, ctx->user_op_conf(),                            \
            job_desc.job_conf().has_cudnn_conv_force_fwd_algo(),                       \
            job_desc.job_conf().cudnn_conv_force_fwd_algo());                          \
      })

REGISTER_CONV_KERNEL(conv1d, float, 1);
REGISTER_CONV_KERNEL(conv2d, float, 2);
REGISTER_CONV_KERNEL(conv3d, float, 3);
REGISTER_CONV_KERNEL(conv1d, double, 1);
REGISTER_CONV_KERNEL(conv2d, double, 2);
REGISTER_CONV_KERNEL(conv3d, double, 3);
REGISTER_CONV_KERNEL(conv1d, float16, 1);
REGISTER_CONV_KERNEL(conv2d, float16, 2);
REGISTER_CONV_KERNEL(conv3d, float16, 3);

template<typename T>
class ConvDataGradGpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvDataGradGpuKernel);
  ConvDataGradGpuKernel() = default;
  ~ConvDataGradGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const JobDesc& job_desc = ctx->job_desc();

    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* filter = ctx->Tensor4ArgNameAndIndex("filter", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    CudnnConvArgsAndAlgo<cudnnConvolutionBwdDataAlgoPerf_t> args_and_algo(
        dx, filter, dy, buf, job_desc, ctx->user_op_conf(), ctx->device_ctx(),
        job_desc.job_conf().has_cudnn_conv_force_bwd_data_algo(),
        job_desc.job_conf().cudnn_conv_force_bwd_data_algo());
    const CudnnConvArgs& args = args_and_algo.args;
    const cudnnConvolutionBwdDataAlgoPerf_t& algo_perf = args_and_algo.algo_perf;

    const void* alpha = CudnnSPOnePtr<T>();
    const void* beta;
    if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), dx->data_type());
      CHECK_EQ(add_to_output->shape(), dx->shape());
      Memcpy<DeviceType::kGPU>(
          ctx->device_ctx(), dx->mut_dptr<void>(), add_to_output->dptr<void>(),
          add_to_output->shape().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      beta = CudnnSPOnePtr<T>();
    } else {
      beta = CudnnSPZeroPtr<T>();
    }

    OF_CUDNN_CHECK(cudnnConvolutionBackwardData(
        ctx->device_ctx()->cudnn_handle(), alpha, args.wdesc.Get(), filter->dptr(),
        args.ydesc.Get(), dy->dptr(), args.cdesc.Get(), algo_perf.algo, buf->mut_dptr(),
        args.params.max_ws_size, beta, args.xdesc.Get(), dx->mut_dptr()));
  }
};

#define REGISTER_CONV_DATA_GRAD_FLOATING_KERNEL(dtype)                                          \
  REGISTER_USER_KERNEL("conv_data_grad")                                                        \
      .SetCreateFn<ConvDataGradGpuKernel<dtype>>()                                              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                       \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))          \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                             \
        const JobDesc& job_desc = *ctx->job_desc();                                             \
        const auto* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);                              \
        const auto* filter = ctx->TensorDesc4ArgNameAndIndex("filter", 0);                      \
        const auto* dx = ctx->TensorDesc4ArgNameAndIndex("dx", 0);                              \
        return InferTmpSizeWithCudnn<cudnnConvolutionBwdDataAlgoPerf_t>(                        \
            dx, filter, dy, job_desc, ctx->user_op_conf(),                                      \
            job_desc.job_conf().has_cudnn_conv_force_bwd_data_algo(),                           \
            job_desc.job_conf().cudnn_conv_force_bwd_data_algo());                              \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext& ctx,                                \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        if (ctx.user_op_conf().has_input("_add_to_output", 0)) {                                \
          OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "_add_to_output", 0, true));          \
        }                                                                                       \
        return Maybe<void>::Ok();                                                               \
      })

REGISTER_CONV_DATA_GRAD_FLOATING_KERNEL(float);
REGISTER_CONV_DATA_GRAD_FLOATING_KERNEL(double);
REGISTER_CONV_DATA_GRAD_FLOATING_KERNEL(float16);

template<typename T>
class ConvFilterGradGpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvFilterGradGpuKernel);
  ConvFilterGradGpuKernel() = default;
  ~ConvFilterGradGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const JobDesc& job_desc = ctx->job_desc();

    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* filter_diff = ctx->Tensor4ArgNameAndIndex("filter_diff", 0);
    user_op::Tensor* buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    CudnnConvArgsAndAlgo<cudnnConvolutionBwdFilterAlgoPerf_t> args_and_algo(
        x, filter_diff, dy, buf, job_desc, ctx->user_op_conf(), ctx->device_ctx(),
        job_desc.job_conf().has_cudnn_conv_force_bwd_filter_algo(),
        job_desc.job_conf().cudnn_conv_force_bwd_filter_algo());
    const CudnnConvArgs& args = args_and_algo.args;
    const cudnnConvolutionBwdFilterAlgoPerf_t& algo_perf = args_and_algo.algo_perf;

    OF_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        ctx->device_ctx()->cudnn_handle(), CudnnSPOnePtr<T>(), args.xdesc.Get(), x->dptr(),
        args.ydesc.Get(), dy->dptr(), args.cdesc.Get(), algo_perf.algo, buf->mut_dptr(),
        args.params.max_ws_size, CudnnSPZeroPtr<T>(), args.wdesc.Get(), filter_diff->mut_dptr()));
  }
};

#define REGISTER_CONV_FILTER_GRAD_FLOATING_KERNEL(dtype)                               \
  REGISTER_USER_KERNEL("conv_filter_grad")                                             \
      .SetCreateFn<ConvFilterGradGpuKernel<dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                    \
        const JobDesc& job_desc = *ctx->job_desc();                                    \
        const auto* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);                     \
        const auto* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);                       \
        const auto* filter_diff = ctx->TensorDesc4ArgNameAndIndex("filter_diff", 0);   \
        return InferTmpSizeWithCudnn<cudnnConvolutionBwdFilterAlgoPerf_t>(             \
            x, filter_diff, dy, job_desc, ctx->user_op_conf(),                         \
            job_desc.job_conf().has_cudnn_conv_force_bwd_filter_algo(),                \
            job_desc.job_conf().cudnn_conv_force_bwd_filter_algo());                   \
      })

REGISTER_CONV_FILTER_GRAD_FLOATING_KERNEL(float);
REGISTER_CONV_FILTER_GRAD_FLOATING_KERNEL(double);
REGISTER_CONV_FILTER_GRAD_FLOATING_KERNEL(float16);

struct ConvBiasGradState final : public user_op::OpKernelState {
  std::unique_ptr<CudnnTensorDesc> bias_diff_desc;
};

template<typename T>
class ConvBiasGradGpuKernel final : public user_op::OpKernel {
 public:
  ConvBiasGradGpuKernel() = default;
  ~ConvBiasGradGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const {
    const auto* bias_diff = ctx->TensorDesc4ArgNameAndIndex("bias_diff", 0);
    const auto* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
    const auto& data_format = ctx->Attr<std::string>("data_format");

    std::shared_ptr<ConvBiasGradState> state(new ConvBiasGradState());
    if (data_format == "channels_first") {
      CHECK_EQ(dy->shape().At(1), bias_diff->shape().At(0));
      state->bias_diff_desc.reset(
          new CudnnTensorDesc(CUDNN_TENSOR_NCHW, bias_diff->data_type(), 1,
                              static_cast<int32_t>(bias_diff->shape().At(0)), 1, 1));
    } else {
      CHECK(data_format == "channels_last") << "Illegal data_format: " << data_format;
      CHECK_EQ(dy->shape().At(dy->shape().NumAxes() - 1), bias_diff->shape().At(0));
      state->bias_diff_desc.reset(
          new CudnnTensorDesc(CUDNN_TENSOR_NHWC, bias_diff->data_type(), 1,
                              static_cast<int32_t>(bias_diff->shape().At(0)), 1, 1));
    }
    return std::move(state);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* bias_diff = ctx->Tensor4ArgNameAndIndex("bias_diff", 0);
    CHECK_EQ(bias_diff->shape().NumAxes(), 1);
    CHECK_GE(dy->shape().NumAxes(), 3);
    CHECK_LE(dy->shape().NumAxes(), 5);

    const std::string& data_format = ctx->Attr<std::string>("data_format");

    std::unique_ptr<CudnnTensorDesc> dy_desc;
    dy_desc.reset(new CudnnTensorDesc(dy->data_type(), dy->shape(), data_format));
    auto* bias_grad_state = dynamic_cast<ConvBiasGradState*>(state);
    CHECK_NOTNULL(bias_grad_state);
    OF_CUDNN_CHECK(cudnnConvolutionBackwardBias(
        ctx->device_ctx()->cudnn_handle(), CudnnSPOnePtr<T>(), dy_desc->Get(), dy->dptr<T>(),
        CudnnSPZeroPtr<T>(), bias_grad_state->bias_diff_desc->Get(), bias_diff->mut_dptr<T>()));
  }
};

#define REGISTER_CONV_BIAS_GRAD_FLOATING_KERNEL(dtype)    \
  REGISTER_USER_KERNEL("conv_bias_grad")                  \
      .SetCreateFn<ConvBiasGradGpuKernel<dtype>>()        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

REGISTER_CONV_BIAS_GRAD_FLOATING_KERNEL(float);
REGISTER_CONV_BIAS_GRAD_FLOATING_KERNEL(double);
REGISTER_CONV_BIAS_GRAD_FLOATING_KERNEL(float16);

}  // namespace

}  // namespace oneflow

#endif
