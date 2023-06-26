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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/user/kernels/fused_clip_grad.h"

namespace oneflow {

namespace {

constexpr int64_t kMultiReduceScaleMulPackSize = 64;

template<typename T>
struct MultiClipGradParamPack {
  MultiClipGradParam<T> params[kMultiReduceScaleMulPackSize];
  size_t size;
};

size_t InferFusedClipGradTempStorageSize(user_op::InferContext* ctx) {
  auto input_size = ctx->input_size("model_diff");
  if (input_size == 0) { return 0; }
  int64_t max_elem_cnt = 0;
  int64_t pack_size = 0;
  int32_t num_blocks = 0;
  for (size_t i = 0; i < input_size; ++i) {
    int64_t elem_cnt = ctx->InputShape("model_diff", i).elem_cnt();
    max_elem_cnt = std::max(max_elem_cnt, elem_cnt);
    pack_size++;
    if (pack_size == kMultiReduceScaleMulPackSize || i == input_size - 1) {
      CHECK_LT(max_elem_cnt, std::numeric_limits<int32_t>::max());
      num_blocks += BlocksNum4ThreadsNum(static_cast<int32_t>(max_elem_cnt));
      max_elem_cnt = 0;
      pack_size = 0;
    }
  }
  CHECK_LT(num_blocks, kCudaThreadsNumPerBlock * kCudaThreadsNumPerBlock * kCudaThreadsNumPerBlock)
      << "Too much blocks needed for computing " << ctx->op_name() << ", should be less than "
      << kCudaThreadsNumPerBlock << "*" << kCudaThreadsNumPerBlock << "*" << kCudaThreadsNumPerBlock
      << ", but got " << num_blocks;
  size_t elem_size = GetSizeOfDataType(ctx->InputDType("model_diff", 0));
  return GetCudaAlignedSize(num_blocks * elem_size * 2);
}

template<typename T>
__global__ void MultiBlockClipGradGpu(MultiClipGradParamPack<T> pack_params, T* scale,
                                      const float norm_type, const float max_norm,
                                      const ClipGradType clip_grad_type,
                                      const bool scale_writable) {
  T t = *scale;
  if (clip_grad_type == ClipGradType::ZeroType) {
    t = static_cast<T>(t > 0);
  } else if (clip_grad_type == ClipGradType::PowerType) {
    t = std::pow(t, 1.f / norm_type);
  }
  if (scale_writable && blockDim.x * blockIdx.x + threadIdx.x == 0) { *scale = t; }
  t = max_norm / (t + 1e-6);
  if (t >= 1.) { return; }
  for (int i = 0; i < pack_params.size; ++i) {
    auto& param = pack_params.params[i];
    CUDA_1D_KERNEL_LOOP(j, param.size) { param.data[j] *= t; }
  }
}

}  // namespace

template<typename T>
struct MultiClipGrad<DeviceType::kCUDA, T> {
  void operator()(ep::Stream* stream, std::vector<MultiClipGradParam<T>>& params, T* scale,
                  const float norm_type, const float max_norm, const ClipGradType clip_grad_type) {
    int32_t total_num_blocks = 0;
    for (size_t i = 0; i < params.size(); i += kMultiReduceScaleMulPackSize) {
      MultiClipGradParamPack<T> pack_params{};
      size_t max_elem_cnt = 0;
      pack_params.size = std::min<size_t>(kMultiReduceScaleMulPackSize, params.size() - i);
      for (size_t j = 0; j < pack_params.size; ++j) {
        pack_params.params[j] = params[i + j];
        max_elem_cnt = std::max<size_t>(max_elem_cnt, pack_params.params[j].size);
      }
      int32_t num_blocks = BlocksNum4ThreadsNum(max_elem_cnt);
      bool scale_writable = static_cast<bool>(i + kMultiReduceScaleMulPackSize >= params.size());
      MultiBlockClipGradGpu<T>
          <<<num_blocks, kCudaThreadsNumPerBlock, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              pack_params, scale, norm_type, max_norm, clip_grad_type, scale_writable);
      total_num_blocks += num_blocks;
    }
  }
};

#define REGISTER_FUSED_CLIP_GRAD_KERNEL(device, dtype)                                         \
  REGISTER_USER_KERNEL("fused_clip_grad")                                                      \
      .SetCreateFn<FusedClipGradKernel<device, dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                    \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))       \
      .SetInferTmpSizeFn(InferFusedClipGradTempStorageSize);

REGISTER_FUSED_CLIP_GRAD_KERNEL(DeviceType::kCUDA, float);
REGISTER_FUSED_CLIP_GRAD_KERNEL(DeviceType::kCUDA, double);

}  // namespace oneflow
