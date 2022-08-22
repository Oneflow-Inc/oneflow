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
/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer
 *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer
 *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */
// modified from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

__device__ __forceinline__ float Add(float* address, float val) { return atomicAdd(address, val); }

__device__ __forceinline__ double Add(double* address, double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  auto address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

template<typename T>
__device__ T DeformableIm2ColBilinear(const T* bottom_data, const int data_width, const int height,
                                      const int width, T h, T w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = bottom_data[h_low * data_width + w_low];
  T v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) v2 = bottom_data[h_low * data_width + w_high];
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) v3 = bottom_data[h_high * data_width + w_low];
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) v4 = bottom_data[h_high * data_width + w_high];

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template<typename T>
__device__ T GetGradientWeight(T argmax_h, T argmax_w, const int h, const int w, const int height,
                               const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;
  if (h == argmax_h_low && w == argmax_w_low) weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high) weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low) weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high) weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template<typename T>
__device__ T GetCoordinateWeight(T argmax_h, T argmax_w, const int height, const int width,
                                 const T* im_data, const int data_width, const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight +=
          -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight +=
          -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template<typename T>
__global__ void DeformableIm2Col(const int nthreads, const T* data_im, const T* data_offset,
                                 const int height, const int width, const int kernel_h,
                                 const int kernel_w, const int pad_h, const int pad_w,
                                 const int stride_h, const int stride_w, const int dilation_h,
                                 const int dilation_w, const int channel_per_deformable_group,
                                 const int batch_size, const int num_channels,
                                 const int deformable_group, const int height_col,
                                 const int width_col, T* data_col) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    T* data_col_ptr =
        data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const T* data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const T* data_offset_ptr = data_offset
                               + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h
                                     * kernel_w * height_col * width_col;

    FOR_RANGE(int, i, 0, kernel_h) {
      FOR_RANGE(int, j, 0, kernel_w) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const T offset_h = data_offset_ptr[data_offset_h_ptr];
        const T offset_w = data_offset_ptr[data_offset_w_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h + offset_h;
        const T w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val = DeformableIm2ColBilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template<typename T>
__global__ void DeformableCol2Im(const int nthreads, const T* data_col, const T* data_offset,
                                 const int channels, const int height, const int width,
                                 const int kernel_h, const int kernel_w, const int pad_h,
                                 const int pad_w, const int stride_h, const int stride_w,
                                 const int dilation_h, const int dilation_w,
                                 const int channel_per_deformable_group, const int batch_size,
                                 const int deformable_group, const int height_col,
                                 const int width_col, T* grad_im) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const T* data_offset_ptr = data_offset
                               + (b * deformable_group + deformable_group_index) * 2 * kernel_h
                                     * kernel_w * height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const T offset_h = data_offset_ptr[data_offset_h_ptr];
    const T offset_w = data_offset_ptr[data_offset_w_ptr];
    const T cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const T cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const T cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 && cur_w + dx < width
            && abs(cur_inv_h_data - (cur_h + dy)) < 1 && abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          T weight = GetGradientWeight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx,
                                       height, width);
          Add(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

template<typename T>
__global__ void DeformableCol2ImCoord(const int nthreads, const T* data_col, const T* data_im,
                                      const T* data_offset, const int channels, const int height,
                                      const int width, const int kernel_h, const int kernel_w,
                                      const int pad_h, const int pad_w, const int stride_h,
                                      const int stride_w, const int dilation_h,
                                      const int dilation_w, const int channel_per_deformable_group,
                                      const int batch_size, const int offset_channels,
                                      const int deformable_group, const int height_col,
                                      const int width_col, T* grad_offset) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    T val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const T* data_col_ptr = data_col
                            + deformable_group_index * channel_per_deformable_group * batch_size
                                  * width_col * height_col;
    const T* data_im_ptr = data_im
                           + (b * deformable_group + deformable_group_index)
                                 * channel_per_deformable_group / kernel_h / kernel_w * height
                                 * width;
    const T* data_offset_ptr = data_offset
                               + (b * deformable_group + deformable_group_index) * 2 * kernel_h
                                     * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step) {
      const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const T offset_h = data_offset_ptr[data_offset_h_ptr];
      const T offset_w = data_offset_ptr[data_offset_w_ptr];
      T inv_h = h_in + i * dilation_h + offset_h;
      T inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) { inv_h = inv_w = -2; }
      const T weight = GetCoordinateWeight(inv_h, inv_w, height, width,
                                           data_im_ptr + cnt * height * width, width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }

    grad_offset[index] = val;
  }
}

}  // namespace

template<typename T>
class DeformableConv2dKernel final : public user_op::OpKernel {
 public:
  DeformableConv2dKernel() = default;
  ~DeformableConv2dKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* offset = ctx->Tensor4ArgNameAndIndex("offset", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ShapeView& input_shape = input->shape();
    const ShapeView& output_shape = output->shape();
    const int64_t out_elem_cnt = output_shape.elem_cnt();
    const int64_t output_bytes = GetCudaAlignedSize(out_elem_cnt * sizeof(T));
    T* column_tmp_buffer = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + output_bytes);
    const int32_t kW = ctx->Attr<int32_t>("kW");
    const int32_t kH = ctx->Attr<int32_t>("kH");
    const int32_t dW = ctx->Attr<int32_t>("dW");
    const int32_t dH = ctx->Attr<int32_t>("dH");
    const int32_t padW = ctx->Attr<int32_t>("padW");
    const int32_t padH = ctx->Attr<int32_t>("padH");
    const int32_t dilationW = ctx->Attr<int32_t>("dilationW");
    const int32_t dilationH = ctx->Attr<int32_t>("dilationH");
    const int32_t group = ctx->Attr<int32_t>("group");
    const int32_t deformable_group = ctx->Attr<int32_t>("deformable_group");
    const int32_t channel_per_deformable_group = input_shape.At(1) / deformable_group;
    const int64_t outputWidth =
        (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    const int64_t outputHeight =
        (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    if (out_elem_cnt > 0) {
      DeformableIm2Col<T><<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0,
                            ctx->device_ctx()->cuda_stream()>>>(
          out_elem_cnt, input->dptr<T>(), offset->dptr<T>(), input_shape.At(2), input_shape.At(3),
          kH, kW, padH, padW, dH, dW, dilationH, dilationW, channel_per_deformable_group,
          input_shape.At(0), input_shape.At(1), deformable_group, output_shape.At(2),
          output_shape.At(3), column_tmp_buffer);
      const int64_t weight_group_offset = weight->shape().elem_cnt() / group;
      const int64_t column_group_offset =
          input_shape.At(1) * kW * kH * input_shape.At(0) * outputHeight * outputWidth / group;
      const int64_t output_group_offset = out_elem_cnt / group;
      FOR_RANGE(int, g, 0, group) {
        NewKernelUtil<DeviceType::kGPU>::OFGemm(
            ctx->device_ctx(), CblasNoTrans, CblasNoTrans, weight->shape().At(0) / group,
            input_shape.At(0) * outputHeight * outputWidth, input_shape.At(1) * kW * kH / group,
            static_cast<T>(1), weight->dptr<T>() + g * weight_group_offset,
            column_tmp_buffer + g * column_group_offset, static_cast<T>(0),
            tmp_buffer->mut_dptr<T>() + g * output_group_offset);
      }
      Shape output_buffer_shape =
          Shape({output_shape.At(1), output_shape.At(0), output_shape.At(2), output_shape.At(3)});
      NewKernelUtil<DeviceType::kGPU>::Transpose(
          ctx->device_ctx(), output_buffer_shape.NumAxes(), output_buffer_shape, output_shape,
          std::vector<int32_t>({1, 0, 2, 3}), output_buffer_shape.elem_cnt(), tmp_buffer->dptr<T>(),
          output->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
template<typename T>
class DeformableConv2dInputGradKernel final : public user_op::OpKernel {
 public:
  DeformableConv2dInputGradKernel() = default;
  ~DeformableConv2dInputGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* output_grad = ctx->Tensor4ArgNameAndIndex("output_grad", 0)
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* offset = ctx->Tensor4ArgNameAndIndex("offset", 0);
    user_op::Tensor* input_grad = ctx->Tensor4ArgNameAndIndex("input_grad", 0);
    user_op::Tensor* offset_grad = ctx->Tensor4ArgNameAndIndex("offset_grad", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ShapeView& output_grad_shape = output_grad->shape();
    const ShapeView& input_shape = input->shape();
    const ShapeView& weight_shape = weight->shape();
    const int32_t kW = ctx->Attr<int32_t>("kW");
    const int32_t kH = ctx->Attr<int32_t>("kH");
    const int32_t dW = ctx->Attr<int32_t>("dW");
    const int32_t dH = ctx->Attr<int32_t>("dH");
    const int32_t padW = ctx->Attr<int32_t>("padW");
    const int32_t padH = ctx->Attr<int32_t>("padH");
    const int32_t dilationW = ctx->Attr<int32_t>("dilationW");
    const int32_t dilationH = ctx->Attr<int32_t>("dilationH");
    const int32_t group = ctx->Attr<int32_t>("group");
    const int32_t deformable_group = ctx->Attr<int32_t>("deformable_group");
    const int32_t channel_per_deformable_group_coord =
        input_shape.At(1) * kH * kW / deformable_group;
    const int32_t channel_per_deformable_group_feat = input_shape.At(1) / deformable_group;
    const int64_t outputWidth =
        (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    const int64_t outputHeight =
        (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
    Memset<DeviceType::kGPU>(ctx->device_ctx(), input_grad->mut_dptr<T>(), 0,
                             input_grad->shape().elem_cnt() * sizeof(T));
    const int64_t nthreads_coord =
        outputHeight * outputWidth * 2 * kH * kW * deformable_group * input_shape.At(0);
    const int64_t nthreads_feat =
        outputHeight * outputWidth * input_shape.At(0) * kH * kW * input_shape.At(1);
    if (nthreads_coord > 0 && nthreads_feat > 0) {
      const int64_t weight_group_offset = weight_shape.elem_cnt() / group;
      const int64_t output_grad_group_offset = output_grad_shape.Count(1) / group;
      const int64_t column_group_offset =
          input_shape.At(1) * kW * kH * input_shape.At(0) * outputHeight * outputWidth / group;
      FOR_RANGE(int, g, 0, group) {
        NewKernelUtil<DeviceType::kGPU>::OFGemm(
            ctx->device_ctx(), CblasTrans, CblasTrans, weight_shape.Count(1),
            input_shape.At(0) * outputHeight * outputWidth, weight_shape.At(0) / group,
            static_cast<T>(1), weight->dptr<T>() + g * weight_group_offset,
            output_grad->dptr<T>() + g * output_grad_group_offset, static_cast<T>(0),
            tmp_buffer->mut_dptr<T>() + g * column_group_offset);
      }
      DeformableCol2ImCoord<T><<<BlocksNum4ThreadsNum(nthreads_coord), kCudaThreadsNumPerBlock, 0,
                                 ctx->device_ctx()->cuda_stream()>>>(
          nthreads_coord, tmp_buffer->dptr<T>(), input->dptr<T>(), offset->dptr<T>(),
          input_shape.At(1), input_shape.At(2), input_shape.At(3), kH, kW, padH, padW, dH, dW,
          dilationH, dilationW, channel_per_deformable_group_coord, input_shape.At(0),
          2 * kH * kW * deformable_group, deformable_group, outputHeight, outputWidth,
          offset_grad->mut_dptr<T>());
      DeformableCol2Im<T><<<BlocksNum4ThreadsNum(nthreads_feat), kCudaThreadsNumPerBlock, 0,
                            ctx->device_ctx()->cuda_stream()>>>(
          nthreads_feat, tmp_buffer->dptr<T>(), offset->dptr<T>(), input_shape.At(1),
          input_shape.At(2), input_shape.At(3), kH, kW, padH, padW, dH, dW, dilationH, dilationW,
          channel_per_deformable_group_feat, input_shape.At(0), deformable_group, outputHeight,
          outputWidth, input_grad->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
template<typename T>
class DeformableConv2dParamGradKernel final : public user_op::OpKernel {
 public:
  DeformableConv2dParamGradKernel() = default;
  ~DeformableConv2dParamGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* output_grad = ctx->Tensor4ArgNameAndIndex("output_grad", 0);
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* offset = ctx->Tensor4ArgNameAndIndex("offset", 0);
    user_op::Tensor* weight_grad = ctx->Tensor4ArgNameAndIndex("weight_grad", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ShapeView& output_grad_shape = output_grad->shape();
    const ShapeView& input_shape = input->shape();
    const int64_t out_elem_cnt = output_grad_shape.elem_cnt();
    const int64_t output_bytes = GetCudaAlignedSize(out_elem_cnt * sizeof(T));
    T* column_tmp_buffer = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + output_bytes);
    const int32_t kW = ctx->Attr<int32_t>("kW");
    const int32_t kH = ctx->Attr<int32_t>("kH");
    const int32_t dW = ctx->Attr<int32_t>("dW");
    const int32_t dH = ctx->Attr<int32_t>("dH");
    const int32_t padW = ctx->Attr<int32_t>("padW");
    const int32_t padH = ctx->Attr<int32_t>("padH");
    const int32_t dilationW = ctx->Attr<int32_t>("dilationW");
    const int32_t dilationH = ctx->Attr<int32_t>("dilationH");
    const int32_t group = ctx->Attr<int32_t>("group");
    const int32_t deformable_group = ctx->Attr<int32_t>("deformable_group");
    const int32_t channel_per_deformable_group = input_shape.At(1) / deformable_group;
    const int64_t outputWidth =
        (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    const int64_t outputHeight =
        (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
    if (out_elem_cnt > 0) {
      DeformableIm2Col<T><<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0,
                            ctx->device_ctx()->cuda_stream()>>>(
          out_elem_cnt, input->dptr<T>(), offset->dptr<T>(), input_shape.At(2), input_shape.At(3),
          kH, kW, padH, padW, dH, dW, dilationH, dilationW, channel_per_deformable_group,
          input_shape.At(0), input_shape.At(1), deformable_group, output_grad_shape.At(2),
          output_grad_shape.At(3), column_tmp_buffer);
      Shape output_grad_buffer_shape = Shape({output_grad_shape.At(1), output_grad_shape.At(0),
                                              output_grad_shape.At(2), output_grad_shape.At(3)});
      NewKernelUtil<DeviceType::kGPU>::Transpose(
          ctx->device_ctx(), output_grad_shape.NumAxes(), output_grad_shape,
          output_grad_buffer_shape, std::vector<int32_t>({1, 0, 2, 3}),
          output_grad_shape.elem_cnt(), output_grad->dptr<T>(), tmp_buffer->mut_dptr<T>());
      const int64_t output_grad_group_offset = output_grad_buffer_shape.elem_cnt() / group;
      const int64_t column_group_offset =
          input_shape.At(1) * kW * kW * input_shape.At(0) * outputHeight * outputWidth / group;
      const int64_t weight_grad_group_offset = weight_grad->shape().elem_cnt() / group;
      FOR_RANGE(int, g, 0, group) {
        NewKernelUtil<DeviceType::kGPU>::OFGemm(
            ctx->device_ctx(), CblasNoTrans, CblasTrans, output_grad_buffer_shape.At(0) / group,
            input_shape.At(1) * kW * kH / group, input_shape.At(0) * outputHeight * outputWidth,
            static_cast<T>(1), tmp_buffer->dptr<T>() + g * output_grad_group_offset,
            column_tmp_buffer + g * column_group_offset, static_cast<T>(0),
            weight_grad->mut_dptr<T>() + g * weight_grad_group_offset);
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_DEFORM_CONV2D_GPU_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("deform_conv2d")                                                            \
      .SetCreateFn<DeformableConv2dKernel<dtype>>()                                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                          \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value))          \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                          \
        const Shape& input_shape = ctx->InputShape("input", 0);                                    \
        Shape* output_shape = ctx->OutputShape("output", 0);                                       \
        const int32_t kW = ctx->Attr<int32_t>("kW");                                               \
        const int32_t kH = ctx->Attr<int32_t>("kH");                                               \
        const int32_t dW = ctx->Attr<int32_t>("dW");                                               \
        const int32_t dH = ctx->Attr<int32_t>("dH");                                               \
        const int32_t padW = ctx->Attr<int32_t>("padW");                                           \
        const int32_t padH = ctx->Attr<int32_t>("padH");                                           \
        const int32_t dilationW = ctx->Attr<int32_t>("dilationW");                                 \
        const int32_t dilationH = ctx->Attr<int32_t>("dilationH");                                 \
        const int64_t outputWidth =                                                                \
            (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;                  \
        const int64_t outputHeight =                                                               \
            (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;                  \
        const int64_t column_bytes =                                                               \
            GetCudaAlignedSize(input_shape.At(1) * kW * kH * input_shape.At(0) * outputHeight      \
                               * outputWidth * sizeof(dtype));                                     \
        const int64_t output_bytes = GetCudaAlignedSize(output_shape->elem_cnt() * sizeof(dtype)); \
        return column_bytes + output_bytes;                                                        \
      });
REGISTER_DEFORM_CONV2D_GPU_KERNEL(float)
REGISTER_DEFORM_CONV2D_GPU_KERNEL(double)
#define REGISTER_DEFORM_CONV2D_INPUT_GRAD_GPU_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("deform_conv2d_input_grad")                                            \
      .SetCreateFn<DeformableConv2dInputGradKernel<dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                     \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)      \
                       & (user_op::HobDataType("weight", 0) == GetDataType<dtype>::value)     \
                       & (user_op::HobDataType("offset", 0) == GetDataType<dtype>::value))    \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                     \
        const Shape& input_shape = ctx->InputShape("input", 0);                               \
        const int32_t kW = ctx->Attr<int32_t>("kW");                                          \
        const int32_t kH = ctx->Attr<int32_t>("kH");                                          \
        const int32_t dW = ctx->Attr<int32_t>("dW");                                          \
        const int32_t dH = ctx->Attr<int32_t>("dH");                                          \
        const int32_t padW = ctx->Attr<int32_t>("padW");                                      \
        const int32_t padH = ctx->Attr<int32_t>("padH");                                      \
        const int32_t dilationW = ctx->Attr<int32_t>("dilationW");                            \
        const int32_t dilationH = ctx->Attr<int32_t>("dilationH");                            \
        const int64_t outputWidth =                                                           \
            (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;             \
        const int64_t outputHeight =                                                          \
            (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;             \
        const int64_t column_bytes =                                                          \
            GetCudaAlignedSize(input_shape.At(1) * kW * kH * input_shape.At(0) * outputHeight \
                               * outputWidth * sizeof(dtype));                                \
        return column_bytes;                                                                  \
      });
REGISTER_DEFORM_CONV2D_INPUT_GRAD_GPU_KERNEL(float)
REGISTER_DEFORM_CONV2D_INPUT_GRAD_GPU_KERNEL(double)
#define REGISTER_DEFORM_CONV2D_PARAM_GRAD_GPU_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("deform_conv2d_param_grad")                                            \
      .SetCreateFn<DeformableConv2dParamGradKernel<dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                     \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)      \
                       & (user_op::HobDataType("offset", 0) == GetDataType<dtype>::value))    \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                     \
        const Shape& input_shape = ctx->InputShape("input", 0);                               \
        const Shape& output_grad_shape = ctx->InputShape("output_grad", 0);                   \
        const int32_t kW = ctx->Attr<int32_t>("kW");                                          \
        const int32_t kH = ctx->Attr<int32_t>("kH");                                          \
        const int32_t dW = ctx->Attr<int32_t>("dW");                                          \
        const int32_t dH = ctx->Attr<int32_t>("dH");                                          \
        const int32_t padW = ctx->Attr<int32_t>("padW");                                      \
        const int32_t padH = ctx->Attr<int32_t>("padH");                                      \
        const int32_t dilationW = ctx->Attr<int32_t>("dilationW");                            \
        const int32_t dilationH = ctx->Attr<int32_t>("dilationH");                            \
        const int64_t outputWidth =                                                           \
            (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;             \
        const int64_t outputHeight =                                                          \
            (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;             \
        const int64_t column_bytes =                                                          \
            GetCudaAlignedSize(input_shape.At(1) * kW * kH * input_shape.At(0) * outputHeight \
                               * outputWidth * sizeof(dtype));                                \
        const int64_t output_bytes =                                                          \
            GetCudaAlignedSize(output_grad_shape.elem_cnt() * sizeof(dtype));                 \
        return column_bytes + output_bytes;                                                   \
      });
REGISTER_DEFORM_CONV2D_PARAM_GRAD_GPU_KERNEL(float)
REGISTER_DEFORM_CONV2D_PARAM_GRAD_GPU_KERNEL(double)
}  // namespace oneflow