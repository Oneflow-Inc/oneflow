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

enum class EncodingLayout {
    kPlanarSinCos,
    kPlanarCosSin,
    kInterleavedSinCos,
    kInterleavedCosSin
};

struct FusedSinusoidalPositionalEncodeParam {
    EncodingLayout layout;
    const void*  in_ptr;
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

template<typename Src>
__global__ void ComputeKernel(struct FusedSinusoidalPositionalEncodeParam param) {
    const Src* in_ptr = reinterpret_cast<const Src*>(param.in_ptr);
    float* out_ptr = param.out_ptr;
    int num_col = param.half_dim * 2;

    if (param.layout == EncodingLayout::kPlanarSinCos) {
        for (int offset = threadIdx.x + blockDim.x * blockIdx.x; offset < param.num_row * num_col; offset += blockDim.x * gridDim.x) {
            float position = in_ptr[offset / num_col];
            int dim = (offset % param.half_dim);
            float exponent = -logf(param.max_period) * dim;
            exponent = exponent / (param.half_dim - param.downscale_freq_shift);
            float emb = expf(exponent) * position * param.scale;

            if ((offset % num_col) < param.half_dim) {
                out_ptr[(offset % num_col) * param.next_stride + 
                    (offset / num_col) * param.stride] = sinf(emb);
            } else {
                out_ptr[(offset % num_col) * param.next_stride + 
                    (offset / num_col) * param.stride] = cosf(emb);
            }
        }

    } else if (param.layout == EncodingLayout::kPlanarCosSin) {
        for (int offset = threadIdx.x + blockDim.x * blockIdx.x; offset < param.num_row * num_col; offset += blockDim.x * gridDim.x) {
            float position = in_ptr[offset / num_col];
            int dim = (offset % param.half_dim);
            float exponent = -logf(param.max_period) * dim;
            exponent = exponent / (param.half_dim - param.downscale_freq_shift);
            float emb = expf(exponent) * position * param.scale;

            if ((offset % num_col) < param.half_dim) {
                out_ptr[(offset % num_col) * param.next_stride + 
                    (offset / num_col) * param.stride] = cosf(emb);
            } else {
                out_ptr[(offset % num_col) * param.next_stride + 
                    (offset / num_col) * param.stride] = sinf(emb);
            }
        }
    } else if (param.layout == EncodingLayout::kInterleavedSinCos) {
        for (int offset = threadIdx.x + blockDim.x * blockIdx.x; offset < param.num_row * num_col; offset += blockDim.x * gridDim.x) {
            float position = in_ptr[offset / num_col];
            int dim = (offset % num_col) / 2;
            float exponent = -logf(param.max_period) * dim;
            exponent = exponent / (param.half_dim - param.downscale_freq_shift);
            float emb = expf(exponent) * position * param.scale;

            if ((offset % 2) == 0) {
                out_ptr[(offset % num_col) * param.next_stride + 
                    (offset / num_col) * param.stride] = sinf(emb);
            } else {
                out_ptr[(offset % num_col) * param.next_stride + 
                    (offset / num_col) * param.stride] = cosf(emb);
            }
        }
    } else if (param.layout == EncodingLayout::kInterleavedCosSin) {
        for (int offset = threadIdx.x + blockDim.x * blockIdx.x; offset < param.num_row * num_col; offset += blockDim.x * gridDim.x) {
            float position = in_ptr[offset / num_col];
            int dim = (offset % num_col) / 2;
            float exponent = -logf(param.max_period) * dim;
            exponent = exponent / (param.half_dim - param.downscale_freq_shift);
            float emb = expf(exponent) * position * param.scale;

            if ((offset % 2) == 0) {
                out_ptr[(offset % num_col) * param.next_stride + 
                    (offset / num_col) * param.stride] = cosf(emb);
            } else {
                out_ptr[(offset % num_col) * param.next_stride + 
                    (offset / num_col) * param.stride] = sinf(emb);
            }
        }
    }

    if (num_col != param.stride) {
        for (int offset = threadIdx.x + blockDim.x * blockIdx.x; offset < param.num_row; offset += blockDim.x * gridDim.x) {
            out_ptr[param.stride * offset + param.stride - 1] = 0.0;
        }
    }
}

void DispatchSrcType(DataType src, struct FusedSinusoidalPositionalEncodeParam& param) {
    if (src == DataType::kInt32) {
        ComputeKernel<int><<<BlocksNum4ThreadsNum(param.num_row * param.half_dim * 2), kCudaThreadsNumPerBlock>>>(param);
    } else if (src == DataType::kFloat) {
        ComputeKernel<float><<<BlocksNum4ThreadsNum(param.num_row * param.half_dim * 2), kCudaThreadsNumPerBlock>>>(param);
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

    const int num_row = positions->shape_view().Count(0, positions->shape_view().NumAxes());
    const int embedding_dim = ctx->Attr<int>("embedding_dim");
    const int half_dim = embedding_dim / 2;
    EncodingLayout layout = static_cast<EncodingLayout>(ctx->Attr<int>("layout"));
    const float downscale_freq_shift = ctx->Attr<float>("downscale_freq_shift");
    const float scale = ctx->Attr<float>("scale");
    const int max_period = ctx->Attr<int>("max_period");

    struct FusedSinusoidalPositionalEncodeParam param = {layout, positions->dptr(), 
        reinterpret_cast<float*>(out->mut_dptr()), num_row, half_dim, 1, 0,
        embedding_dim, downscale_freq_shift, scale, max_period};

    DispatchSrcType(positions->data_type(), param);
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
