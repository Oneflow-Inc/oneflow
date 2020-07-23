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
#include <iostream>

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
__device__ __forceinline__ int32_t GetOffset(int32_t n, int32_t h, int32_t w, int32_t c, int32_t H,
                                             int32_t W, int32_t C);

template<>
__device__ __forceinline__ int32_t GetOffset<TensorLayout::kNCHW>(int32_t n, int32_t h, int32_t w,
                                                                  int32_t c, int32_t H, int32_t W,
                                                                  int32_t C) {
  return n * C * H * W + c * H * W + h * W + w;  //  n,c,h,w
}

template<>
__device__ __forceinline__ int32_t GetOffset<TensorLayout::kNHWC>(int32_t n, int32_t h, int32_t w,
                                                                  int32_t c, int32_t H, int32_t W,
                                                                  int32_t C) {
  return n * H * W * C + h * W * C + w * C + c;  //  n,h,w,c
}

template<TensorLayout layout>
__global__ void TransposeMirrorNormalizeGpuImpl(int32_t elem_cnt, const uint8_t* x_dptr,
                                                float* y_dptr, int32_t N, int32_t H, int32_t W,
                                                int32_t C, const int8_t* mirror,
                                                const NormalizeVal mean,
                                                const NormalizeVal inv_std) {
  int32_t x_stride_n = H * W * C;
  int32_t x_stride_h = W * C;
  int32_t x_stride_w = C;
  CUDA_1D_KERNEL_LOOP(x_idx, elem_cnt) {
    int32_t x = x_idx;
    int32_t n = x / x_stride_n;
    x %= x_stride_n;
    int32_t h = x / x_stride_h;
    x %= x_stride_h;
    int32_t w = x / x_stride_w;
    int32_t c = x % x_stride_w;
    assert(n >= 0 && n < N);
    assert(h >= 0 && h < H);
    assert(w >= 0 && w < W);
    assert(c >= 0 && c < C);
    if (mirror && mirror[n]) { w = W - 1 - w; }
    float mean_val = mean.val[c];
    float inv_std_val = inv_std.val[c];
    int32_t y_idx = GetOffset<layout>(n, h, w, c, H, W, C);
    y_dptr[y_idx] = (static_cast<float>(x_dptr[x_idx]) - mean_val) * inv_std_val;
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
    const int8_t* mirror_dptr = nullptr;
    user_op::Tensor* mirror_blob = ctx->Tensor4ArgNameAndIndex("mirror", 0);
    if (mirror_blob) { mirror_dptr = mirror_blob->dptr<int8_t>(); }

    if (output_layout == "NCHW") {
      TransposeMirrorNormalizeGpuImpl<TensorLayout::kNCHW>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
             ctx->device_ctx()->cuda_stream()>>>(elem_cnt, in_dptr, out_dptr, N, H, W, C,
                                                 mirror_dptr, mean, inv_std);
    } else if (output_layout == "NHWC") {
      TransposeMirrorNormalizeGpuImpl<TensorLayout::kNHWC>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
             ctx->device_ctx()->cuda_stream()>>>(elem_cnt, in_dptr, out_dptr, N, H, W, C,
                                                 mirror_dptr, mean, inv_std);
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
