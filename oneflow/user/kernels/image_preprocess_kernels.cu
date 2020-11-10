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
#include "oneflow/core/common/fixed_vector.h"
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
__device__ __forceinline__ void OutIdx2InIdx(int32_t* out_idx, int32_t* in_idx,
                                             const int8_t* mirror_dptr, int32_t out_W,
                                             int32_t H_offset, int32_t W_offset);
template<>
__device__ __forceinline__ void OutIdx2InIdx<TensorLayout::kNCHW>(int32_t* out_idx, int32_t* in_idx,
                                                                  const int8_t* mirror_dptr,
                                                                  int32_t out_W, int32_t H_offset,
                                                                  int32_t W_offset) {
  if (mirror_dptr && mirror_dptr[out_idx[0]]) { out_idx[3] = out_W - 1 - out_idx[3]; }
  in_idx[0] = out_idx[0];             // N
  in_idx[1] = out_idx[2] + H_offset;  // H
  in_idx[2] = out_idx[3] + W_offset;  // W
  in_idx[3] = out_idx[1];             // C
}

template<>
__device__ __forceinline__ void OutIdx2InIdx<TensorLayout::kNHWC>(int32_t* out_idx, int32_t* in_idx,
                                                                  const int8_t* mirror_dptr,
                                                                  int32_t out_W, int32_t H_offset,
                                                                  int32_t W_offset) {
  if (mirror_dptr && mirror_dptr[out_idx[0]]) { out_idx[2] = out_W - 1 - out_idx[2]; }
  in_idx[0] = out_idx[0];             // N
  in_idx[1] = out_idx[1] + H_offset;  // H
  in_idx[2] = out_idx[2] + W_offset;  // W
  in_idx[3] = out_idx[3];             // C
}

template<TensorLayout layout>
__global__ void CropMirrorNormalizeGpuImpl(int32_t elem_cnt, const uint8_t* in_dptr,
                                           float* out_dptr, const int8_t* mirror_dptr,
                                           int32_t out_W,
                                           const NdIndexOffsetHelper<int32_t, 4> in_helper,
                                           const NdIndexOffsetHelper<int32_t, 4> out_helper,
                                           int32_t H_offset, int32_t W_offset,
                                           const NormalizeVal mean, const NormalizeVal inv_std) {
  CUDA_1D_KERNEL_LOOP(out_offset, elem_cnt) {
    int32_t in_idx[4];
    int32_t out_idx[4];
    out_helper.OffsetToNdIndex(out_offset, out_idx);
    OutIdx2InIdx<layout>(out_idx, in_idx, mirror_dptr, out_W, H_offset, W_offset);
    float mean_val;
    float inv_std_val;
    const int32_t c = in_idx[3];
    // When the compiler can't resolve array indices to constants it will put private arrays into
    // GPU local memory. Using local memory is slower than keeping array elements directly in
    // registers.
    if (c == 0) {
      mean_val = mean.val[0];
      inv_std_val = inv_std.val[0];
    } else if (c == 1) {
      mean_val = mean.val[1];
      inv_std_val = inv_std.val[1];
    } else if (c == 2) {
      mean_val = mean.val[2];
      inv_std_val = inv_std.val[2];
    } else {
      // undefined behavior
      assert(false);
    }
    int32_t in_offset = in_helper.NdIndexToOffset(in_idx);
    out_dptr[out_offset] = (static_cast<float>(in_dptr[in_offset]) - mean_val) * inv_std_val;
  }
}

}  // namespace

class CropMirrorNormalizeGpuKernel final : public user_op::OpKernel {
 public:
  CropMirrorNormalizeGpuKernel() = default;
  ~CropMirrorNormalizeGpuKernel() override = default;

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
    const ShapeView& out_shape = out_blob->shape();
    CHECK_EQ(in_shape.NumAxes(), 4);
    CHECK_EQ(out_shape.NumAxes(), 4);
    int32_t elem_cnt = out_shape.elem_cnt();
    CHECK_LE(elem_cnt, GetMaxVal<int32_t>());
    float crop_pos_y = ctx->Attr<float>("crop_pos_y");
    float crop_pos_x = ctx->Attr<float>("crop_pos_x");

    int32_t N = in_shape.At(0);
    int32_t in_H = in_shape.At(1);
    int32_t in_W = in_shape.At(2);
    int32_t C = in_shape.At(3);
    const NdIndexOffsetHelper<int32_t, 4> in_helper(N, in_H, in_W, C);
    const int8_t* mirror_dptr = nullptr;
    user_op::Tensor* mirror_blob = ctx->Tensor4ArgNameAndIndex("mirror", 0);
    if (mirror_blob) { mirror_dptr = mirror_blob->dptr<int8_t>(); }

    if (output_layout == "NCHW") {
      CHECK_EQ(N, out_shape.At(0));
      CHECK_EQ(C, out_shape.At(1));
      int32_t out_H = out_shape.At(2);
      int32_t out_W = out_shape.At(3);
      CHECK_LE(out_H, in_H);
      CHECK_LE(out_W, in_W);
      int32_t H_offset = (in_H - out_H) * crop_pos_y;
      int32_t W_offset = (in_W - out_W) * crop_pos_x;
      const NdIndexOffsetHelper<int32_t, 4> out_helper(N, C, out_H, out_W);
      CropMirrorNormalizeGpuImpl<TensorLayout::kNCHW>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
             ctx->device_ctx()->cuda_stream()>>>(elem_cnt, in_dptr, out_dptr, mirror_dptr, out_W,
                                                 in_helper, out_helper, H_offset, W_offset, mean,
                                                 inv_std);
    } else if (output_layout == "NHWC") {
      CHECK_EQ(N, out_shape.At(0));
      int32_t out_H = out_shape.At(1);
      int32_t out_W = out_shape.At(2);
      CHECK_EQ(C, out_shape.At(3));
      CHECK_LE(out_H, in_H);
      CHECK_LE(out_W, in_W);
      int32_t H_offset = (in_H - out_H) * crop_pos_y;
      int32_t W_offset = (in_W - out_W) * crop_pos_x;
      const NdIndexOffsetHelper<int32_t, 4> out_helper(N, out_H, out_W, C);
      CropMirrorNormalizeGpuImpl<TensorLayout::kNHWC>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
             ctx->device_ctx()->cuda_stream()>>>(elem_cnt, in_dptr, out_dptr, mirror_dptr, out_W,
                                                 in_helper, out_helper, H_offset, W_offset, mean,
                                                 inv_std);
    } else {
      UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("crop_mirror_normalize_from_uint8")
    .SetCreateFn<CropMirrorNormalizeGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("in", 0) == DataType::kUInt8)
                     & (user_op::HobDataType("out", 0) == DataType::kFloat));

}  // namespace oneflow
