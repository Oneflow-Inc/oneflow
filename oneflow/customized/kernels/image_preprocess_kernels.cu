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
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

struct NormalizeVal {
  float val[3];
};

enum TensorLayout {
  kNCHW = 0,
  kNHWC = 1,
};

class NormalizeAttr final : public user_op::OpKernelState {
 public:
  NormalizeAttr(user_op::KernelInitContext* ctx) {
    const std::vector<float>& mean_vec = ctx->Attr<std::vector<float>>("mean");
    if (mean_vec.size() == 1) {
      for (int i = 0; i < 3; ++i) { mean_.val[i] = mean_vec.at(0); }
    } else if (mean_vec.size() == 3) {
      for (int i = 0; i < 3; ++i) { mean_.val[i] = mean_vec.at(i); }
    } else {
      UNIMPLEMENTED();
    }

    const std::vector<float>& std_vec = ctx->Attr<std::vector<float>>("std");
    if (std_vec.size() == 1) {
      for (int i = 0; i < 3; ++i) { inv_std_.val[i] = 1.0f / std_vec.at(0); }
    } else if (std_vec.size() == 3) {
      for (int i = 0; i < 3; ++i) { inv_std_.val[i] = 1.0f / std_vec.at(i); }
    } else {
      UNIMPLEMENTED();
    }
  }
  ~NormalizeAttr() = default;

  const NormalizeVal& mean() const { return mean_; }
  const NormalizeVal& inv_std() const { return inv_std_; }

 private:
  NormalizeVal mean_;
  NormalizeVal inv_std_;
};

template<TensorLayout layout>
__device__ __forceinline__ int32_t GetYOffset(const int32_t* x_idx,
                                              const NdIndexOffsetHelper<int32_t, 4> y_helper);

template<>
__device__ __forceinline__ int32_t GetYOffset<TensorLayout::kNCHW>(
    const int32_t* x_idx, const NdIndexOffsetHelper<int32_t, 4> y_helper) {
  return y_helper.NdIndexToOffset(*x_idx, *(x_idx + 3), *(x_idx + 1), *(x_idx + 2));
}

template<>
__device__ __forceinline__ int32_t GetYOffset<TensorLayout::kNHWC>(
    const int32_t* x_idx, const NdIndexOffsetHelper<int32_t, 4> y_helper) {
  return y_helper.NdIndexToOffset(x_idx);
}

template<TensorLayout layout>
__global__ void TransposeMirrorNormalizeGpuImpl(int32_t elem_cnt, const uint8_t* x_dptr,
                                                float* y_dptr, const int8_t* mirror_dptr, int32_t W,
                                                const NdIndexOffsetHelper<int32_t, 4> x_helper,
                                                const NdIndexOffsetHelper<int32_t, 4> y_helper,
                                                const NormalizeVal mean,
                                                const NormalizeVal inv_std) {
  CUDA_1D_KERNEL_LOOP(x_offset, elem_cnt) {
    int32_t x_idx[4];
    x_helper.OffsetToNdIndex(x_offset, x_idx);
    if (mirror_dptr && mirror_dptr[x_idx[0]]) { x_idx[2] = W - 1 - x_idx[2]; }
    float mean_val = mean.val[x_idx[3]];
    float inv_std_val = inv_std.val[x_idx[3]];
    int32_t y_offset = GetYOffset<layout>(x_idx, y_helper);
    y_dptr[y_offset] = (static_cast<float>(x_dptr[x_offset]) - mean_val) * inv_std_val;
  }
}

}  // namespace

class TransposeMirrorNormalizeGpuKernel final : public user_op::OpKernel {
 public:
  TransposeMirrorNormalizeGpuKernel() = default;
  ~TransposeMirrorNormalizeGpuKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NormalizeAttr>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* normalize_attr = dynamic_cast<NormalizeAttr*>(state);
    const NormalizeVal& mean = normalize_attr->mean();
    const NormalizeVal& inv_std = normalize_attr->inv_std();
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    const std::string& output_layout = ctx->Attr<std::string>("output_layout");
    float* out_dptr = out_blob->mut_dptr<float>();

    const uint8_t* in_dptr = in_blob->dptr<uint8_t>();
    const ShapeView& in_shape = in_blob->shape();
    CHECK_EQ(in_shape.NumAxes(), 4);
    int32_t elem_cnt = in_shape.elem_cnt();
    CHECK_LE(elem_cnt, GetMaxVal<int32_t>());

    int32_t N = in_shape.At(0);
    int32_t H = in_shape.At(1);
    int32_t W = in_shape.At(2);
    int32_t C = in_shape.At(3);
    const NdIndexOffsetHelper<int32_t, 4> x_helper(N, H, W, C);
    const int8_t* mirror_dptr = nullptr;
    user_op::Tensor* mirror_blob = ctx->Tensor4ArgNameAndIndex("mirror", 0);
    if (mirror_blob) { mirror_dptr = mirror_blob->dptr<int8_t>(); }

    if (output_layout == "NCHW") {
      const NdIndexOffsetHelper<int32_t, 4> y_helper(N, C, H, W);
      TransposeMirrorNormalizeGpuImpl<TensorLayout::kNCHW>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
             ctx->device_ctx()->cuda_stream()>>>(elem_cnt, in_dptr, out_dptr, mirror_dptr, W,
                                                 x_helper, y_helper, mean, inv_std);
    } else if (output_layout == "NHWC") {
      const NdIndexOffsetHelper<int32_t, 4> y_helper(N, H, W, C);
      TransposeMirrorNormalizeGpuImpl<TensorLayout::kNHWC>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
             ctx->device_ctx()->cuda_stream()>>>(elem_cnt, in_dptr, out_dptr, mirror_dptr, W,
                                                 x_helper, y_helper, mean, inv_std);
    } else {
      UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("transpose_mirror_normalize_gpu")
    .SetCreateFn<TransposeMirrorNormalizeGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU)
                     & (user_op::HobDataType("in", 0) == DataType::kUInt8)
                     & (user_op::HobDataType("out", 0) == DataType::kFloat));

}  // namespace oneflow
