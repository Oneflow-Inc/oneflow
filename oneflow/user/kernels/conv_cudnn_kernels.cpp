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
#include "oneflow/core/device/cudnn_conv_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/job/lazy_mode.h"
#if defined(USE_HIP)
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#endif  // defined(USE_HIP)

namespace oneflow {

namespace {

template<typename PerfT>
struct CudnnConvArgsAndAlgo final {
  using AlgoT = decltype(std::declval<PerfT>().algo);

  CudnnConvArgs args;
  PerfT algo_perf;

  CudnnConvArgsAndAlgo(const user_op::Tensor* x, const user_op::Tensor* w, const user_op::Tensor* y,
                       user_op::Tensor* buf, const user_op::KernelComputeContext* ctx,
                       ep::Stream* stream, bool has_forced_algo, int32_t forced_algo)
      : args(*ctx, x->data_type(), x->shape_view(), w->data_type(), w->shape_view(), y->data_type(),
             y->shape_view(), ctx->Attr<std::string>("data_format"), buf->shape_view().elem_cnt(),
             Singleton<ResourceDesc, ForSession>::Get()
                     ->resource()
                     .cudnn_conf()
                     .cudnn_conv_heuristic_search_algo()
                 || (!LazyMode::is_enabled()),
             Singleton<ResourceDesc, ForSession>::Get()
                 ->resource()
                 .cudnn_conf()
                 .cudnn_conv_use_deterministic_algo_only(),
             Singleton<ResourceDesc, ForSession>::Get()
                     ->resource()
                     .cudnn_conf()
                     .cudnn_conv_enable_pseudo_half()
                 || (ctx->Attr<std::string>("data_format") == "channels_last"
                     && std::is_same<PerfT, cudnnConvolutionBwdFilterAlgoPerf_t>::value)) {
    size_t byte_size_of_buf = buf->shape_view().elem_cnt();
    AllocatedCudnnConvResource res(stream->As<ep::CudaStream>()->cudnn_handle(),
                                   const_cast<void*>(x->dptr()), const_cast<void*>(w->dptr()),
                                   const_cast<void*>(y->dptr()), buf->mut_dptr());
    if (has_forced_algo) {
      algo_perf = GetCudnnConvAlgorithmPerferenceWithResource<PerfT>(
          &args, &res, static_cast<AlgoT>(forced_algo));
    } else {
      algo_perf = FindCudnnConvAlgorithmWithResource<PerfT>(&args, &res);
    }
    CHECK_EQ(algo_perf.status, CUDNN_STATUS_SUCCESS)
        << "op (" << ctx->op_name()
        << ") find algorithm perference failed. algo: " << algo_perf.algo;
    CHECK_LE(algo_perf.memory, byte_size_of_buf)
        << "op (" << ctx->op_name() << ") find algorithm " << algo_perf.algo << ", need memory "
        << algo_perf.memory << ", but cudnn_buf_limit_byte is " << byte_size_of_buf;
    OF_CUDNN_CHECK(cudnnSetConvolutionMathType(args.cdesc.Get(), algo_perf.mathType));
  }
  CudnnConvArgsAndAlgo() = delete;
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvArgsAndAlgo);
};

template<typename PerfT>
size_t InferTmpSizeWithCudnn(const user_op::TensorDesc* x, const user_op::TensorDesc* w,
                             const user_op::TensorDesc* y, const user_op::InferContext& ctx,
                             bool has_forced_algo, int32_t forced_algo) {
  using AlgoT = decltype(std::declval<PerfT>().algo);

  const auto& cudnn_conf = Singleton<ResourceDesc, ForSession>::Get()->resource().cudnn_conf();
  size_t workspace_size = cudnn_conf.cudnn_buf_limit_mbyte() * 1024 * 1024;
  if (!x->is_dynamic()) {
    CudnnConvArgs args(ctx, x->data_type(), ShapeView(x->shape()), w->data_type(),
                       ShapeView(w->shape()), y->data_type(), ShapeView(y->shape()),
                       ctx.Attr<std::string>("data_format"), workspace_size,
                       cudnn_conf.cudnn_conv_heuristic_search_algo() || (!LazyMode::is_enabled()),
                       cudnn_conf.cudnn_conv_use_deterministic_algo_only(),
                       cudnn_conf.cudnn_conv_enable_pseudo_half()
                           || (ctx.Attr<std::string>("data_format") == "channels_last"
                               && std::is_same<PerfT, cudnnConvolutionBwdFilterAlgoPerf_t>::value));
    PerfT algo_perf{};
    if (has_forced_algo) {
      algo_perf = GetCudnnConvAlgorithmPerference<PerfT>(&args, static_cast<AlgoT>(forced_algo));
    } else {
      algo_perf = FindCudnnConvAlgorithm<PerfT>(&args);
    }
    CHECK_EQ(algo_perf.status, CUDNN_STATUS_SUCCESS)
        << "op (" << ctx.op_name()
        << ") find algorithm perference failed. algo: " << algo_perf.algo;
    CHECK_LE(algo_perf.memory, workspace_size)
        << "op (" << ctx.op_name() << ") find algorithm " << algo_perf.algo << ", need memory "
        << algo_perf.memory << ", but cudnn_buf_limit_byte is " << workspace_size;
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

struct ConvCudnnOpKernelCache final : public user_op::OpKernelCache {
  std::unique_ptr<CudnnTensorDesc> bias_desc;
};

template<size_t NDims>
class ConvGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ConvGpuKernel() = default;
  ~ConvGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<ConvCudnnOpKernelCache> CreateConvCudnnOpKernelCache(
      user_op::KernelCacheContext* ctx) const {
    const auto& data_format = ctx->Attr<std::string>("data_format");
    int32_t filters = ctx->Attr<int32_t>("filters");

    std::shared_ptr<ConvCudnnOpKernelCache> state(new ConvCudnnOpKernelCache());

    const user_op::TensorDesc* bias = ctx->TensorDesc4ArgNameAndIndex("bias", 0);
    if (bias != nullptr) {
      state->bias_desc.reset(
          GetBiasCudnnTensorDesc<NDims>(data_format, filters, bias->data_type()));
    }

    return state;
  }

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateConvCudnnOpKernelCache(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    if (in->shape_view().elem_cnt() == 0) return;
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto& cudnn_conf = Singleton<ResourceDesc, ForSession>::Get()->resource().cudnn_conf();
    CudnnConvArgsAndAlgo<cudnnConvolutionFwdAlgoPerf_t> args_and_algo(
        in, weight, out, buf, ctx, ctx->stream(), cudnn_conf.has_cudnn_conv_force_fwd_algo(),
        cudnn_conf.cudnn_conv_force_fwd_algo());
    const CudnnConvArgs& args = args_and_algo.args;
    const cudnnConvolutionFwdAlgoPerf_t& algo_perf = args_and_algo.algo_perf;
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);

    const void* beta = nullptr;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), out->data_type());
      CHECK_EQ(add_to_output->shape_view(), out->shape_view());
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), out->mut_dptr(), add_to_output->dptr(),
          add_to_output->shape_view().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      beta = CudnnSPOnePtr(in->data_type());
    } else {
      beta = CudnnSPZeroPtr(in->data_type());
    }

    OF_CUDNN_CHECK(cudnnConvolutionForward(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CudnnSPOnePtr(in->data_type()),
        args.xdesc.Get(), in->dptr(), args.wdesc.Get(), weight->dptr(), args.cdesc.Get(),
        algo_perf.algo, buf->mut_dptr(), args.params.max_ws_size, beta, args.ydesc.Get(),
        out->mut_dptr()));

    if (bias != nullptr) {
      const auto* conv_cache = dynamic_cast<const ConvCudnnOpKernelCache*>(cache);
      CHECK_NOTNULL(conv_cache);
      OF_CUDNN_CHECK(cudnnAddTensor(ctx->stream()->As<ep::CudaStream>()->cudnn_handle(),
                                    CudnnSPOnePtr(in->data_type()), conv_cache->bias_desc->Get(),
                                    bias->dptr(), CudnnSPOnePtr(in->data_type()), args.ydesc.Get(),
                                    out->mut_dptr()));
    }
  }

  bool IsCudaGraphSupported(user_op::KernelInitContext* ctx,
                            user_op::OpKernelState* state) const override {
    return Singleton<ResourceDesc, ForSession>::Get()
        ->resource()
        .cudnn_conf()
        .cudnn_conv_heuristic_search_algo();
  }
};

#define REGISTER_CONV_KERNEL(op_name, ndims)                                                \
  REGISTER_USER_KERNEL(#op_name)                                                            \
      .SetCreateFn<ConvGpuKernel<ndims>>()                                                  \
      .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA)                       \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                         \
        const auto& in = ctx->InputTensorDesc("in", 0);                                     \
        if (in.shape().elem_cnt() == 0) return 0;                                           \
        const auto& weight = ctx->InputTensorDesc("weight", 0);                             \
        const auto& out = ctx->OutputTensorDesc("out", 0);                                  \
        const auto& cudnn_conf =                                                            \
            Singleton<ResourceDesc, ForSession>::Get()->resource().cudnn_conf();            \
        return InferTmpSizeWithCudnn<cudnnConvolutionFwdAlgoPerf_t>(                        \
            &in, &weight, &out, *ctx, cudnn_conf.has_cudnn_conv_force_fwd_algo(),           \
            cudnn_conf.cudnn_conv_force_fwd_algo());                                        \
      })                                                                                    \
      .SetInplaceProposalFn(                                                                \
          [](const user_op::InferContext& ctx,                                              \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {        \
            if (ctx.has_input("_add_to_output", 0)) {                                       \
              OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "_add_to_output", 0, true)); \
            }                                                                               \
            return Maybe<void>::Ok();                                                       \
          });

REGISTER_CONV_KERNEL(conv1d, 1);
REGISTER_CONV_KERNEL(conv2d, 2);
REGISTER_CONV_KERNEL(conv3d, 3);

class ConvDataGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvDataGradGpuKernel);
  ConvDataGradGpuKernel() = default;
  ~ConvDataGradGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* filter = ctx->Tensor4ArgNameAndIndex("filter", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx->shape_view().elem_cnt() == 0) return;
    user_op::Tensor* buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto& cudnn_conf = Singleton<ResourceDesc, ForSession>::Get()->resource().cudnn_conf();

    CudnnConvArgsAndAlgo<cudnnConvolutionBwdDataAlgoPerf_t> args_and_algo(
        dx, filter, dy, buf, ctx, ctx->stream(), cudnn_conf.has_cudnn_conv_force_bwd_data_algo(),
        cudnn_conf.cudnn_conv_force_bwd_data_algo());
    const CudnnConvArgs& args = args_and_algo.args;
    const cudnnConvolutionBwdDataAlgoPerf_t& algo_perf = args_and_algo.algo_perf;

    const void* alpha = CudnnSPOnePtr(dy->data_type());
    const void* beta = nullptr;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), dx->data_type());
      CHECK_EQ(add_to_output->shape_view(), dx->shape_view());
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), dx->mut_dptr<void>(), add_to_output->dptr<void>(),
          add_to_output->shape_view().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      beta = CudnnSPOnePtr(dy->data_type());
    } else {
      beta = CudnnSPZeroPtr(dy->data_type());
    }

    OF_CUDNN_CHECK(cudnnConvolutionBackwardData(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), alpha, args.wdesc.Get(),
        filter->dptr(), args.ydesc.Get(), dy->dptr(), args.cdesc.Get(), algo_perf.algo,
        buf->mut_dptr(), args.params.max_ws_size, beta, args.xdesc.Get(), dx->mut_dptr()));
  }

  bool IsCudaGraphSupported(user_op::KernelInitContext* ctx,
                            user_op::OpKernelState* state) const override {
    return Singleton<ResourceDesc, ForSession>::Get()
        ->resource()
        .cudnn_conf()
        .cudnn_conv_heuristic_search_algo();
  }
};

REGISTER_USER_KERNEL("conv_data_grad")
    .SetCreateFn<ConvDataGradGpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA)
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {
      const auto& dy = ctx->InputTensorDesc("dy", 0);
      const auto& filter = ctx->InputTensorDesc("filter", 0);
      const auto& dx = ctx->OutputTensorDesc("dx", 0);
      if (dx.shape().elem_cnt() == 0) return 0;
      const auto& cudnn_conf = Singleton<ResourceDesc, ForSession>::Get()->resource().cudnn_conf();
      return InferTmpSizeWithCudnn<cudnnConvolutionBwdDataAlgoPerf_t>(
          &dx, &filter, &dy, *ctx, cudnn_conf.has_cudnn_conv_force_bwd_data_algo(),
          cudnn_conf.cudnn_conv_force_bwd_data_algo());
    })
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.has_input("_add_to_output", 0)) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "_add_to_output", 0, true));
      }
      return Maybe<void>::Ok();
    });

class ConvFilterGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvFilterGradGpuKernel);
  ConvFilterGradGpuKernel() = default;
  ~ConvFilterGradGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* filter_diff = ctx->Tensor4ArgNameAndIndex("filter_diff", 0);
    if (x->shape_view().elem_cnt() == 0) {
      Memset<DeviceType::kCUDA>(
          ctx->stream(), filter_diff->mut_dptr(), 0,
          filter_diff->shape_view().elem_cnt() * GetSizeOfDataType(filter_diff->data_type()));
      return;
    }
    user_op::Tensor* buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto& cudnn_conf = Singleton<ResourceDesc, ForSession>::Get()->resource().cudnn_conf();

    CudnnConvArgsAndAlgo<cudnnConvolutionBwdFilterAlgoPerf_t> args_and_algo(
        x, filter_diff, dy, buf, ctx, ctx->stream(),
        cudnn_conf.has_cudnn_conv_force_bwd_filter_algo(),
        cudnn_conf.cudnn_conv_force_bwd_filter_algo());
    const CudnnConvArgs& args = args_and_algo.args;
    const cudnnConvolutionBwdFilterAlgoPerf_t& algo_perf = args_and_algo.algo_perf;

    OF_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CudnnSPOnePtr(dy->data_type()),
        args.xdesc.Get(), x->dptr(), args.ydesc.Get(), dy->dptr(), args.cdesc.Get(), algo_perf.algo,
        buf->mut_dptr(), args.params.max_ws_size, CudnnSPZeroPtr(dy->data_type()), args.wdesc.Get(),
        filter_diff->mut_dptr()));
  }

  bool IsCudaGraphSupported(user_op::KernelInitContext* ctx,
                            user_op::OpKernelState* state) const override {
    return Singleton<ResourceDesc, ForSession>::Get()
        ->resource()
        .cudnn_conf()
        .cudnn_conv_heuristic_search_algo();
  }
};

REGISTER_USER_KERNEL("conv_filter_grad")
    .SetCreateFn<ConvFilterGradGpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA)
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {
      const auto& dy = ctx->InputTensorDesc("dy", 0);
      const auto& x = ctx->InputTensorDesc("x", 0);
      if (x.shape().elem_cnt() == 0) return 0;
      const auto& filter_diff = ctx->OutputTensorDesc("filter_diff", 0);
      const auto& cudnn_conf = Singleton<ResourceDesc, ForSession>::Get()->resource().cudnn_conf();
      return InferTmpSizeWithCudnn<cudnnConvolutionBwdFilterAlgoPerf_t>(
          &x, &filter_diff, &dy, *ctx, cudnn_conf.has_cudnn_conv_force_bwd_filter_algo(),
          cudnn_conf.cudnn_conv_force_bwd_filter_algo());
    });

struct ConvBiasGradState final : public user_op::OpKernelState {
  std::unique_ptr<CudnnTensorDesc> bias_diff_desc;
};

#if defined(USE_HIP)
// Byte offsets into the channels_first bias gradient's scratch: ones over the spatial axis in
// dy's type, then, in the accumulation type, ones over the samples, the per-(sample, channel)
// sums and the unrounded gradient.
struct ConvBiasGradHipLayout {
  ConvBiasGradHipLayout(DataType data_type, int64_t num, int64_t channels, int64_t spatial) {
    const size_t acc_size =
        GetSizeOfDataType(data_type == DataType::kDouble ? DataType::kDouble : DataType::kFloat);
    num_ones = GetCudaAlignedSize(spatial * GetSizeOfDataType(data_type));
    row_sums = num_ones + GetCudaAlignedSize(num * acc_size);
    bias_diff = row_sums + GetCudaAlignedSize(num * channels * acc_size);
    size = bias_diff + GetCudaAlignedSize(channels * acc_size);
  }
  size_t num_ones;
  size_t row_sums;
  size_t bias_diff;
  size_t size;
};

#endif  // defined(USE_HIP)
class ConvBiasGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ConvBiasGradGpuKernel() = default;
  ~ConvBiasGradGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<ConvBiasGradState> CreateConvBiasGradState(
      user_op::KernelComputeContext* ctx) const {
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
    return state;
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* bias_diff = ctx->Tensor4ArgNameAndIndex("bias_diff", 0);
    CHECK_EQ(bias_diff->shape_view().NumAxes(), 1);
    CHECK_GE(dy->shape_view().NumAxes(), 3);
    CHECK_LE(dy->shape_view().NumAxes(), 5);

    const std::string& data_format = ctx->Attr<std::string>("data_format");

#if defined(USE_HIP)
    // MIOpen no longer implements bias backward (miopenConvolutionBackwardBias returns
    // miopenStatusNotImplemented), so the bias gradient is dy times a vector of ones.
    const ShapeView& shape = dy->shape_view();
    const bool channels_first = (data_format == "channels_first");
    const int64_t num = shape.At(0);
    const int64_t channels = channels_first ? shape.At(1) : shape.At(shape.NumAxes() - 1);
    const int64_t spatial = channels_first ? shape.Count(2) : shape.Count(1, shape.NumAxes() - 1);
    const DataType data_type = dy->data_type();
    auto fill =
        ep::primitive::NewPrimitive<ep::primitive::FillFactory>(DeviceType::kCUDA, data_type);
    CHECK(fill);
    if (num * spatial == 0) {
      fill->Launch(ctx->stream(), bias_diff->mut_dptr(), Scalar(0), channels);
      return;
    }
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    if (channels_first) {
      // Sum every (sample, channel) row of dy over the spatial axis into a float (double for
      // double dy) scratch, then sum the rows over the samples, so that a float16 or bfloat16
      // gradient is accumulated in float and rounded once.
      const DataType acc_type =
          data_type == DataType::kDouble ? DataType::kDouble : DataType::kFloat;
      const ConvBiasGradHipLayout layout(data_type, num, channels, spatial);
      char* spatial_ones = tmp_buffer->mut_dptr<char>();
      char* num_ones = spatial_ones + layout.num_ones;
      char* row_sums = spatial_ones + layout.row_sums;
      fill->Launch(ctx->stream(), spatial_ones, Scalar(1), spatial);
      auto acc_fill =
          ep::primitive::NewPrimitive<ep::primitive::FillFactory>(DeviceType::kCUDA, acc_type);
      CHECK(acc_fill);
      acc_fill->Launch(ctx->stream(), num_ones, Scalar(1), num);
      const auto GetHipDataType = [](DataType type) {
        switch (type) {
          case DataType::kFloat16: return CUDA_R_16F;
          case DataType::kBFloat16: return CUDA_R_16BF;
          case DataType::kFloat: return CUDA_R_32F;
          case DataType::kDouble: return CUDA_R_64F;
          default: UNIMPLEMENTED(); return CUDA_R_32F;
        }
      };
      const int64_t rows = num * channels;
      CHECK_LE(rows, std::numeric_limits<int>::max());
      CHECK_LE(spatial, std::numeric_limits<int>::max());
      const float f_one = 1;
      const float f_zero = 0;
      const double d_one = 1;
      const double d_zero = 0;
      const bool is_double = acc_type == DataType::kDouble;
      OF_CUBLAS_CHECK(cublasGemmEx(
          ctx->stream()->As<ep::CudaStream>()->cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
          static_cast<int>(rows), 1, static_cast<int>(spatial),
          is_double ? static_cast<const void*>(&d_one) : static_cast<const void*>(&f_one),
          dy->dptr(), GetHipDataType(data_type), static_cast<int>(spatial), spatial_ones,
          GetHipDataType(data_type), static_cast<int>(spatial),
          is_double ? static_cast<const void*>(&d_zero) : static_cast<const void*>(&f_zero),
          row_sums, GetHipDataType(acc_type), static_cast<int>(rows),
          is_double ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      auto matmul = ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(
          DeviceType::kCUDA, acc_type, ep::primitive::BlasTransposeType::T,
          ep::primitive::BlasTransposeType::N);
      CHECK(matmul);
      // row_sums is num x channels.
      void* acc_bias_diff =
          acc_type == data_type ? bias_diff->mut_dptr() : spatial_ones + layout.bias_diff;
      matmul->Launch(ctx->stream(), channels, 1, num, Scalar(1), row_sums, num_ones, Scalar(0),
                     acc_bias_diff);
      if (acc_type != data_type) {
        auto cast = ep::primitive::NewPrimitive<ep::primitive::CastFactory>(DeviceType::kCUDA,
                                                                            acc_type, data_type);
        CHECK(cast);
        cast->Launch(ctx->stream(), acc_bias_diff, bias_diff->mut_dptr(), channels);
      }
    } else {
      // dy is (num * spatial) x channels.
      fill->Launch(ctx->stream(), tmp_buffer->mut_dptr(), Scalar(1), num * spatial);
      auto matmul = ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(
          DeviceType::kCUDA, data_type, ep::primitive::BlasTransposeType::T,
          ep::primitive::BlasTransposeType::N);
      CHECK(matmul);
      matmul->Launch(ctx->stream(), channels, 1, num * spatial, Scalar(1), dy->dptr(),
                     tmp_buffer->dptr(), Scalar(0), bias_diff->mut_dptr());
    }
#else
    std::unique_ptr<CudnnTensorDesc> dy_desc;
    dy_desc.reset(new CudnnTensorDesc(dy->data_type(), dy->shape_view(), data_format));
    const auto& bias_grad_state = CreateConvBiasGradState(ctx);
    CHECK_NOTNULL(bias_grad_state.get());
    OF_CUDNN_CHECK(cudnnConvolutionBackwardBias(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CudnnSPOnePtr(dy->data_type()),
        dy_desc->Get(), dy->dptr(), CudnnSPZeroPtr(dy->data_type()),
        bias_grad_state->bias_diff_desc->Get(), bias_diff->mut_dptr()));
#endif  // defined(USE_HIP)
  }
};

REGISTER_USER_KERNEL("conv_bias_grad")
    .SetCreateFn<ConvBiasGradGpuKernel>()
#if defined(USE_HIP)
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {
      const Shape& shape = ctx->InputTensorDesc("dy", 0).shape();
      const DataType data_type = ctx->InputTensorDesc("dy", 0).data_type();
      if (ctx->Attr<std::string>("data_format") == "channels_first") {
        return ConvBiasGradHipLayout(data_type, shape.At(0), shape.At(1), shape.Count(2)).size;
      }
      return shape.Count(0, shape.NumAxes() - 1) * GetSizeOfDataType(data_type);
    })
#endif  // defined(USE_HIP)
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA);

}  // namespace

}  // namespace oneflow

#endif
