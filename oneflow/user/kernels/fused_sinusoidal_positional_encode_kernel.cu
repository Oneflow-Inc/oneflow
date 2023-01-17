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

struct FusedSinusoidalPositionalEncodeParam {
    const void*  in_ptr;
    float* out_ptr;
    int N;
    int embedding_dim;
    int next_stride;
    int init_offset;
    int stride;
    float downscale_freq_shift;
    float scale;
    int max_period;
};

enum class EncodingPattern {
    SIN_COS,
    COS_SIN,
    INTERLEAVED_SIN_COS,
    INTERLEAVED_COS_SIN
};

template<typename Src>
__global__ void ComputeCosKernel(struct FusedSinusoidalPositionalEncodeParam param) {
    const Src* in_ptr = reinterpret_cast<const Src*>(param.in_ptr);
    float* out_ptr = param.out_ptr;

    for (int offset = threadIdx.x + blockDim.x * blockIdx.x; offset < param.N * param.embedding_dim; offset += blockDim.x * gridDim.x) {
        float position = in_ptr[offset / param.embedding_dim];
        int dim = (offset % param.embedding_dim);
        float exponent = -logf(param.max_period) * dim;
        exponent = exponent / (param.embedding_dim - param.downscale_freq_shift);
        float emb = expf(exponent) * position * param.scale;

        out_ptr[(offset % param.embedding_dim) * param.next_stride + 
            (offset / param.embedding_dim) * param.stride] = cosf(emb);
    }
}

template<typename Src>
__global__ void ComputeSinKernel(struct FusedSinusoidalPositionalEncodeParam param) {
    const Src* in_ptr = reinterpret_cast<const Src*>(param.in_ptr);
    float* out_ptr = param.out_ptr;

    for (int offset = threadIdx.x + blockDim.x * blockIdx.x; offset < param.N * param.embedding_dim; offset += blockDim.x * gridDim.x) {
        float position = in_ptr[offset / param.embedding_dim];
        int dim = (offset % param.embedding_dim);
        float exponent = -logf(param.max_period) * dim;
        exponent = exponent / (param.embedding_dim - param.downscale_freq_shift);
        float emb = expf(exponent) * position * param.scale;

        out_ptr[(offset % param.embedding_dim) * param.next_stride + 
            (offset / param.embedding_dim) * param.stride] = sinf(emb);
    }
}

__global__ void PaddingKernel(float* out_ptr, int N, int embedding_dim) {
    for (int offset = threadIdx.x + blockDim.x * blockIdx.x; offset < N; offset += blockDim.x * gridDim.x) {
        out_ptr[embedding_dim * offset + embedding_dim - 1] = 0.0;
    }
}

class FusedSinusoidalPositionalEncodeKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  FusedSinusoidalPositionalEncodeKernel() = default;
  ~FusedSinusoidalPositionalEncodeKernel() override = default;


 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    const user_op::Tensor* positions = ctx->Tensor4ArgNameAndIndex("positions", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("encoded_positions", 0);

    const int N = positions->shape_view().At(0);
    const int embedding_dim = ctx->Attr<int>("embedding_dim");
    const int half_dim = embedding_dim / 2;
    EncodingPattern pattern = static_cast<EncodingPattern>(ctx->Attr<int>("pattern")); //TODO: should be four different types
    const float downscale_freq_shift = ctx->Attr<float>("downscale_freq_shift");
    const float scale = ctx->Attr<float>("scale");
    const int max_period = ctx->Attr<int>("max_period");

    struct FusedSinusoidalPositionalEncodeParam sin_param = {positions->dptr(), 
        reinterpret_cast<float*>(out->mut_dptr()), N, half_dim, 1, 0,
        embedding_dim, downscale_freq_shift, scale, max_period};
    struct FusedSinusoidalPositionalEncodeParam cos_param = {positions->dptr(), 
        reinterpret_cast<float*>(out->mut_dptr()) + half_dim, N, half_dim, 1, half_dim,
        embedding_dim, downscale_freq_shift, scale, max_period};

    const int num_threads = 256;
    const int num_blocks = MIN((N * half_dim + num_threads - 1) / num_threads, 8192);

    if (pattern == EncodingPattern::SIN_COS) {
        // do nothing
    } else if (pattern == EncodingPattern::COS_SIN) {
        cos_param.out_ptr = reinterpret_cast<float*>(out->mut_dptr());
        sin_param.out_ptr = reinterpret_cast<float*>(out->mut_dptr()) + half_dim;
        cos_param.init_offset = 0;
        sin_param.init_offset = half_dim;
    } else if (pattern == EncodingPattern::INTERLEAVED_SIN_COS) {
        sin_param.out_ptr = reinterpret_cast<float*>(out->mut_dptr());
        cos_param.out_ptr = reinterpret_cast<float*>(out->mut_dptr()) + 1;
        sin_param.next_stride = 2;
        cos_param.next_stride = 2;
    } else if (pattern == EncodingPattern::INTERLEAVED_COS_SIN) {
        cos_param.out_ptr = reinterpret_cast<float*>(out->mut_dptr());
        sin_param.out_ptr = reinterpret_cast<float*>(out->mut_dptr()) + 1;
        sin_param.next_stride = 2;
        cos_param.next_stride = 2;
        cos_param.init_offset = half_dim;
        sin_param.init_offset = 0;
    } else {
        //TODO: alarm
    }

    //TODO: Dispatch Src
    ComputeSinKernel<int><<<num_blocks, num_threads>>>(sin_param);
    ComputeCosKernel<int><<<num_blocks, num_threads>>>(cos_param);

    if (embedding_dim % 2 == 1) {
        PaddingKernel<<<(N + 255) / 256, 256>>>(reinterpret_cast<float*>(out->mut_dptr()), N, embedding_dim);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_SINUSOIDAL_POSITIONAL_ENCODE_KERNEL(data_type)               \
  REGISTER_USER_KERNEL("fused_sinusoidal_positional_encode")                            \
      .SetCreateFn<FusedSinusoidalPositionalEncodeKernel>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("positions", 0) == data_type));

REGISTER_FUSED_SINUSOIDAL_POSITIONAL_ENCODE_KERNEL(DataType::kInt32);
REGISTER_FUSED_SINUSOIDAL_POSITIONAL_ENCODE_KERNEL(DataType::kFloat);

}  // namespace

}  // namespace oneflow
