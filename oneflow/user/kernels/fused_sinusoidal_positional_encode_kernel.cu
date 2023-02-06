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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

enum class EncodingLayout { kPlanarSinCos, kPlanarCosSin, kInterleavedSinCos, kInterleavedCosSin };

struct FusedSinusoidalPositionalEncodeParam {
  ep::Stream* stream;
  const void* in_ptr;
  float* out_ptr;
  int num_row;
  int half_dim;
  int next_stride;  // stride of next element
  int init_offset;  // offset of the first element
  int stride;       // offset of next row
  float downscale_freq_shift;
  float scale;
  int max_period;
};

template<EncodingLayout layout>
__device__ auto InferFunc(int offset, int num_col, int half_dim) {
  bool callSin;

  if (layout == EncodingLayout::kPlanarSinCos) {
    callSin = ((offset % num_col) < half_dim);
  } else if (layout == EncodingLayout::kPlanarCosSin) {
    callSin = ((offset % num_col) >= half_dim);
  } else if (layout == EncodingLayout::kInterleavedSinCos) {
    callSin = ((offset % 2) == 0);
  } else if (layout == EncodingLayout::kInterleavedCosSin) {
    callSin = ((offset % 2) == 1);
  }

  if (callSin) {
    return sinf;
  } else {
    return cosf;
  }
}

template<EncodingLayout layout>
__device__ int InferDim(int offset, int num_col, int half_dim) {
  int dim;

  if (layout == EncodingLayout::kPlanarSinCos || layout == EncodingLayout::kPlanarCosSin) {
    dim = (offset % half_dim);
  } else if (layout == EncodingLayout::kInterleavedSinCos || layout == EncodingLayout::kInterleavedCosSin) {
    dim = ((offset % num_col) / 2);
  }

  return dim;
}

template<typename Src, EncodingLayout layout>
__global__ void ComputeKernel(struct FusedSinusoidalPositionalEncodeParam param) {
  const Src* in_ptr = reinterpret_cast<const Src*>(param.in_ptr);
  float* out_ptr = param.out_ptr;
  int num_col = param.half_dim * 2;

  for (int offset = threadIdx.x + blockDim.x * blockIdx.x; offset < param.num_row * num_col;
        offset += blockDim.x * gridDim.x) {
    float position = in_ptr[offset / num_col];
    int dim = InferDim<layout>(offset, num_col, param.half_dim);
    float exponent = -logf(param.max_period) * dim;
    exponent = exponent / (param.half_dim - param.downscale_freq_shift);
    float emb = expf(exponent) * position * param.scale;

    auto func = InferFunc<layout>(offset, num_col, param.half_dim);
    out_ptr[(offset % num_col) * param.next_stride + (offset / num_col) * param.stride] = func(emb);
  }

  if (param.stride % 2 != 0) {
    for (int offset = threadIdx.x + blockDim.x * blockIdx.x; offset < param.num_row;
         offset += blockDim.x * gridDim.x) {
      out_ptr[param.stride * offset + param.stride - 1] = 0.0;
    }
  }
}

template<typename Src>
void DispatchLayout(EncodingLayout layout, struct FusedSinusoidalPositionalEncodeParam& param) {
  if (layout == EncodingLayout::kPlanarSinCos) {
    ComputeKernel<Src, EncodingLayout::kPlanarSinCos>
        <<<BlocksNum4ThreadsNum(param.num_row * param.half_dim * 2), kCudaThreadsNumPerBlock, 0,
           param.stream->As<ep::CudaStream>()->cuda_stream()>>>(param);
  } else if (layout == EncodingLayout::kPlanarCosSin) {
    ComputeKernel<Src, EncodingLayout::kPlanarCosSin>
        <<<BlocksNum4ThreadsNum(param.num_row * param.half_dim * 2), kCudaThreadsNumPerBlock, 0,
           param.stream->As<ep::CudaStream>()->cuda_stream()>>>(param);
  } else if (layout == EncodingLayout::kInterleavedSinCos) {
    ComputeKernel<Src, EncodingLayout::kInterleavedSinCos>
        <<<BlocksNum4ThreadsNum(param.num_row * param.half_dim * 2), kCudaThreadsNumPerBlock, 0,
           param.stream->As<ep::CudaStream>()->cuda_stream()>>>(param);
  } else if (layout == EncodingLayout::kInterleavedCosSin) {
    ComputeKernel<Src, EncodingLayout::kInterleavedCosSin>
        <<<BlocksNum4ThreadsNum(param.num_row * param.half_dim * 2), kCudaThreadsNumPerBlock, 0,
           param.stream->As<ep::CudaStream>()->cuda_stream()>>>(param);
  }
}

void DispatchSrcType(DataType src, EncodingLayout layout,
                     struct FusedSinusoidalPositionalEncodeParam& param) {
  if (src == DataType::kInt32) {
    DispatchLayout<int>(layout, param);
  } else if (src == DataType::kFloat) {
    DispatchLayout<float>(layout, param);
  }
}

class FusedSinusoidalPositionalEncodeKernel final : public user_op::OpKernel,
                                                    public user_op::CudaGraphSupport {
 public:
  FusedSinusoidalPositionalEncodeKernel() = default;
  ~FusedSinusoidalPositionalEncodeKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* positions = ctx->Tensor4ArgNameAndIndex("positions", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("encoded_positions", 0);

    const int num_row = positions->shape_view().Count(0, positions->shape_view().NumAxes());
    const int embedding_dim = ctx->Attr<int>("embedding_dim");
    const int half_dim = embedding_dim / 2;
    EncodingLayout layout = static_cast<EncodingLayout>(ctx->Attr<int>("layout"));
    const float downscale_freq_shift = ctx->Attr<float>("downscale_freq_shift");
    const float scale = ctx->Attr<float>("scale");
    const int max_period = ctx->Attr<int>("max_period");

    struct FusedSinusoidalPositionalEncodeParam param = {ctx->stream(),
                                                         positions->dptr(),
                                                         reinterpret_cast<float*>(out->mut_dptr()),
                                                         num_row,
                                                         half_dim,
                                                         1,
                                                         0,
                                                         embedding_dim,
                                                         downscale_freq_shift,
                                                         scale,
                                                         max_period};

    DispatchSrcType(positions->data_type(), layout, param);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_SINUSOIDAL_POSITIONAL_ENCODE_KERNEL(data_type)  \
  REGISTER_USER_KERNEL("fused_sinusoidal_positional_encode")           \
      .SetCreateFn<FusedSinusoidalPositionalEncodeKernel>()            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("positions", 0) == data_type));

REGISTER_FUSED_SINUSOIDAL_POSITIONAL_ENCODE_KERNEL(DataType::kInt32);
REGISTER_FUSED_SINUSOIDAL_POSITIONAL_ENCODE_KERNEL(DataType::kFloat);

}  // namespace

}  // namespace oneflow
