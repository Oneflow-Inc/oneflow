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
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {
namespace {

template<typename PerfT>
struct CudnnDeConvArgsAndAlgo final {
  using AlgoT = decltype(std::declval<PerfT>().algo);

  CudnnConvArgs args;
  PerfT algo_perf;

  // TODO(hanbinbin): remove arg job_desc and set cudnn_conv config as args of
  // CudnnDeConvArgsAndAlgo
  CudnnDeConvArgsAndAlgo(const user_op::Tensor* x, const user_op::Tensor* w,
                         const user_op::Tensor* y, user_op::Tensor* buf,
                         const user_op::KernelComputeContext* ctx, DeviceCtx* device_ctx,
                         bool has_forced_algo, int32_t forced_algo)
      : args(*ctx, x->data_type(), x->shape(), w->data_type(), w->shape(), y->data_type(),
             y->shape(), ctx->Attr<std::string>("data_format"), buf->shape().elem_cnt(),
             Global<ResourceDesc, ForSession>::Get()
                 ->resource()
                 .cudnn_conf()
                 .cudnn_conv_heuristic_search_algo(),
             Global<ResourceDesc, ForSession>::Get()
                 ->resource()
                 .cudnn_conf()
                 .cudnn_conv_use_deterministic_algo_only(),
             Global<ResourceDesc, ForSession>::Get()
                 ->resource()
                 .cudnn_conf()
                 .cudnn_conv_enable_pseudo_half()) {
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
        << "op (" << ctx->op_name()
        << ") find algorithm perference failed. algo: " << algo_perf.algo;
    CHECK_LE(algo_perf.memory, byte_size_of_buf)
        << "op (" << ctx->op_name() << ") find algorithm " << algo_perf.algo << ", need memory "
        << algo_perf.memory << ", but cudnn_buf_limit_byte is " << byte_size_of_buf;
  }
  CudnnDeConvArgsAndAlgo() = delete;
  OF_DISALLOW_COPY_AND_MOVE(CudnnDeConvArgsAndAlgo);
};

template<typename PerfT>
size_t InferTmpSizeWithCudnn(const user_op::TensorDesc* x, const user_op::TensorDesc* w,
                             const user_op::TensorDesc* y, const user_op::InferContext& ctx,
                             bool has_forced_algo, int32_t forced_algo) {
  using AlgoT = decltype(std::declval<PerfT>().algo);

  const auto& cudnn_conf = Global<ResourceDesc, ForSession>::Get()->resource().cudnn_conf();
  size_t workspace_size = cudnn_conf.cudnn_buf_limit_mbyte() * 1024 * 1024;
  if (!x->is_dynamic()) {
    CudnnConvArgs args(ctx, x->data_type(), ShapeView(x->shape()), w->data_type(),
                       ShapeView(w->shape()), y->data_type(), ShapeView(y->shape()),
                       ctx.Attr<std::string>("data_format"), workspace_size,
                       cudnn_conf.cudnn_conv_heuristic_search_algo(),
                       cudnn_conf.cudnn_conv_use_deterministic_algo_only(),
                       cudnn_conf.cudnn_conv_enable_pseudo_half());
    PerfT algo_perf;
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

}  // namespace

template<typename T, size_t NDims>
class DeConvGpuKernel final : public user_op::OpKernel {
 public:
  DeConvGpuKernel() = default;
  ~DeConvGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto& cudnn_conf = Global<ResourceDesc, ForSession>::Get()->resource().cudnn_conf();

    CudnnDeConvArgsAndAlgo<cudnnConvolutionBwdDataAlgoPerf_t> args_and_algo(
        out, weight, in, buf, ctx, ctx->device_ctx(),
        cudnn_conf.has_cudnn_conv_force_bwd_data_algo(),
        cudnn_conf.cudnn_conv_force_bwd_data_algo());
    const CudnnConvArgs& args = args_and_algo.args;
    const cudnnConvolutionBwdDataAlgoPerf_t& algo_perf = args_and_algo.algo_perf;

    OF_CUDNN_CHECK(cudnnConvolutionBackwardData(
        ctx->device_ctx()->cudnn_handle(), CudnnSPOnePtr<T>(), args.wdesc.Get(), weight->dptr(),
        args.ydesc.Get(), in->dptr(), args.cdesc.Get(), algo_perf.algo, buf->mut_dptr(),
        args.params.max_ws_size, CudnnSPZeroPtr<T>(), args.xdesc.Get(), out->mut_dptr()));
  }
};

#define REGISTER_DECONV_KERNEL(op_name, dtype, ndims)                                              \
  REGISTER_USER_KERNEL(#op_name)                                                                   \
      .SetCreateFn<DeConvGpuKernel<dtype, ndims>>()                                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                          \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))             \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                                \
        const auto* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);                                 \
        const auto* weight = ctx->TensorDesc4ArgNameAndIndex("weight", 0);                         \
        const auto* out = ctx->OutputTensorDesc("out", 0);                                         \
        const auto& cudnn_conf = Global<ResourceDesc, ForSession>::Get()->resource().cudnn_conf(); \
        return InferTmpSizeWithCudnn<cudnnConvolutionBwdDataAlgoPerf_t>(                           \
            out, weight, in, *ctx, cudnn_conf.has_cudnn_conv_force_bwd_data_algo(),                \
            cudnn_conf.cudnn_conv_force_bwd_data_algo());                                          \
      })

REGISTER_DECONV_KERNEL(deconv1d, float, 1);
REGISTER_DECONV_KERNEL(deconv2d, float, 2);
REGISTER_DECONV_KERNEL(deconv3d, float, 3);
REGISTER_DECONV_KERNEL(deconv1d, double, 1);
REGISTER_DECONV_KERNEL(deconv2d, double, 2);
REGISTER_DECONV_KERNEL(deconv3d, double, 3);

}  // namespace oneflow

#endif
