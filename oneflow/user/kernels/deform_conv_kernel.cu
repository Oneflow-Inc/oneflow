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

#include "oneflow/core/framework/user_op_hob.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/include/primitive/memset.h"

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
__device__ T bilinear_interpolate(const T* in, int height, int width, T h, T w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) { return 0; }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = in[h_low * width + w_low];
  T v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) v2 = in[h_low * width + w_high];
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) v3 = in[h_high * width + w_low];
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) v4 = in[h_high * width + w_high];

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
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
__device__ T get_coordinate_weight(const T* im_data, int height, int width, T y, T x,
                                   bool is_y_direction) {
  int y_l = floor(y);
  int x_l = floor(x);
  int y_h = y_l + 1;
  int x_h = x_l + 1;

  bool valid_y_l = 0 <= y_l && y_l < height;
  bool valid_y_h = 0 <= y_h && y_h < height;
  bool valid_x_l = 0 <= x_l && x_l < width;
  bool valid_x_h = 0 <= x_h && x_h < width;

  T zero = 0;
  T v_yx = (valid_y_l && valid_x_l) ? im_data[y_l * width + x_l] : zero;
  T v_yX = (valid_y_l && valid_x_h) ? im_data[y_l * width + x_h] : zero;
  T v_Yx = (valid_y_h && valid_x_l) ? im_data[y_h * width + x_l] : zero;
  T v_YX = (valid_y_h && valid_x_h) ? im_data[y_h * width + x_h] : zero;

  if (is_y_direction) {
    T dx = x - x_l;
    return dx * (v_YX - v_yX) + (1 - dx) * (v_Yx - v_yx);
  } else {
    T dy = y - y_l;
    return dy * (v_YX - v_Yx) + (1 - dy) * (v_yX - v_yx);
  }
}

template<typename T>
__device__ T GetGradientWeight(T argmax_h, T argmax_w, const int h, const int w, const int height,
                               const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
    // empty
    return static_cast<T>(0);
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
__global__ void DeformableCol2Im(int n, const T* col, const T* offset_data, const T* mask_data,
                                 int channels, int height, int width, int kernel_h, int kernel_w,
                                 int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                                 int dilation_w, int batch_sz, int n_offset_grps, int out_h,
                                 int out_w, bool use_mask, T* grad_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int b = (index / (out_w * out_h)) % batch_sz;
    const int j = (index / (out_w * out_h * batch_sz)) % kernel_w;
    const int i = (index / (out_w * out_h * batch_sz * kernel_w)) % kernel_h;
    const int c = index / (out_w * out_h * batch_sz * kernel_w * kernel_h);

    int c_per_offset_grp = channels / n_offset_grps;
    const int offset_grp = c / c_per_offset_grp;
    auto offset_ptr = offset_data;

    offset_ptr += (b * n_offset_grps + offset_grp) * 2 * kernel_h * kernel_w * out_h * out_w;
    auto mask_ptr = mask_data;
    if (use_mask) {
      mask_ptr += (b * n_offset_grps + offset_grp) * kernel_h * kernel_w * out_h * out_w;
    }

    const int mask_idx = i * kernel_w + j;
    const int offset_idx = 2 * mask_idx;

    const int offset_h_ptr = ((offset_idx)*out_h + out_y) * out_w + out_x;
    const int offset_w_ptr = ((offset_idx + 1) * out_h + out_y) * out_w + out_x;

    const T offset_h = offset_ptr[offset_h_ptr];
    const T offset_w = offset_ptr[offset_w_ptr];

    T mask_value = 1;
    if (use_mask) { mask_value = mask_ptr[(mask_idx * out_h + out_y) * out_w + out_x]; }

    const T y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
    const T x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        int yp = (int)y + dy;
        int xp = (int)x + dx;
        if (0 <= yp && yp < height && 0 <= xp && xp < width && abs(y - yp) < 1 && abs(x - xp) < 1) {
          int grad_pos = ((b * channels + c) * height + yp) * width + xp;
          T weight = (1 - abs(y - yp)) * (1 - abs(x - xp));
          Add(grad_im + grad_pos, mask_value * weight * col[index]);
        }
      }
    }
  }
}

template<typename T>
__global__ void DeformableIm2Col(int n, const T* input, const T* offset, const T* mask, int height,
                                 int width, int weight_h, int weight_w, int pad_h, int pad_w,
                                 int stride_h, int stride_w, int dilation_h, int dilation_w,
                                 int batch_sz, int n_in_channels, int n_offset_grps, int out_h,
                                 int out_w, bool use_mask, T* columns) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int out_b = (index / (out_w * out_h)) % batch_sz;
    const int in_c = index / (out_w * out_h * batch_sz);
    const int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    const int grp_idx = in_c / c_per_offset_grp;
    auto columns_ptr = columns;
    columns_ptr +=
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) + out_y * out_w + out_x);
    auto input_ptr = input;
    input_ptr += (out_b * (n_in_channels * height * width) + in_c * (height * width));
    auto offset_ptr = offset;
    offset_ptr += (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w * out_h * out_w;
    auto mask_ptr = mask;
    if (use_mask) {
      mask_ptr += (out_b * n_offset_grps + grp_idx) * weight_h * weight_w * out_h * out_w;
    }

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const int mask_idx = i * weight_w + j;
        const int offset_idx = 2 * mask_idx;

        T mask_value = 1;
        if (use_mask) { mask_value = mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x]; }

        const T offset_h = offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
        const T offset_w = offset_ptr[(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];
        const T y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
        const T x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;
        *columns_ptr = mask_value * bilinear_interpolate(input_ptr, height, width, y, x);
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }
}

template<typename T>
__global__ void DeformableCol2imCoord(int n, const T* col_data, const T* im_data,
                                      const T* offset_data, const T* mask_data, int channels,
                                      int height, int width, int weight_h, int weight_w, int pad_h,
                                      int pad_w, int stride_h, int stride_w, int dilation_h,
                                      int dilation_w, int batch_sz, int offset_channels,
                                      int n_offset_grps, int out_h, int out_w, const bool use_mask,
                                      T* grad_offset, T* grad_mask) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    T grad_offset_val = 0;
    T grad_mask_val = 0;

    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int w_w = (index / (out_w * out_h * 2)) % weight_w;
    int w_h = (index / (out_w * out_h * 2 * weight_w)) % weight_h;
    int c = (index / (out_w * out_h)) % offset_channels;
    int b = index / (out_w * out_h * offset_channels);

    const int offset_grp = c / (2 * weight_h * weight_w);
    const int col_step = weight_h * weight_w;

    int c_per_offset_grp = channels / n_offset_grps;
    auto col_ptr = col_data;
    col_ptr += offset_grp * c_per_offset_grp * weight_h * weight_w * batch_sz * out_w * out_h;
    auto im_ptr = im_data;
    im_ptr += (b * n_offset_grps + offset_grp) * c_per_offset_grp * height * width;
    auto offset_ptr = offset_data;
    offset_ptr += (b * n_offset_grps + offset_grp) * 2 * weight_h * weight_w * out_h * out_w;
    auto mask_ptr = mask_data;
    if (use_mask) {
      mask_ptr += (b * n_offset_grps + offset_grp) * weight_h * weight_w * out_h * out_w;
    }

    const int offset_c = c - offset_grp * 2 * weight_h * weight_w;
    const bool is_y_direction = offset_c % 2 == 0;

    const int c_bound = c_per_offset_grp * weight_h * weight_w;
    for (int col_c = (offset_c / 2); col_c < c_bound; col_c += col_step) {
      const int col_pos = (((col_c * batch_sz + b) * out_h) + h) * out_w + w;

      int out_x = col_pos % out_w;
      int out_y = (col_pos / out_w) % out_h;
      int j = (col_pos / (out_w * out_h * batch_sz)) % weight_w;
      int i = (col_pos / (out_w * out_h * batch_sz * weight_w)) % weight_h;

      const int mask_idx = i * weight_w + j;

      const int offset_h_ptr = (((2 * mask_idx) * out_h + out_y) * out_w + out_x);
      const int offset_w_ptr = (((2 * mask_idx + 1) * out_h + out_y) * out_w + out_x);
      const T offset_h = offset_ptr[offset_h_ptr];
      const T offset_w = offset_ptr[offset_w_ptr];

      T mask_value = 1;
      if (use_mask) { mask_value = mask_ptr[(mask_idx * out_h + out_y) * out_w + out_x]; }

      T y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
      T x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

      const T weight = get_coordinate_weight(im_ptr, height, width, y, x, is_y_direction);
      grad_offset_val += mask_value * weight * col_ptr[col_pos];

      if (use_mask && is_y_direction) {
        grad_mask_val += col_ptr[col_pos] * bilinear_interpolate(im_ptr, height, width, y, x);
      }

      im_ptr += height * width;
    }

    grad_offset[index] = grad_offset_val;

    if (use_mask && is_y_direction) {
      const int idx =
          ((((b * n_offset_grps + offset_grp) * weight_h + w_h) * weight_w + w_w) * out_h + h)
              * out_w
          + w;
      grad_mask[idx] = grad_mask_val;
    }
  }
}

}  // namespace

ep::primitive::BlasTransposeType GetBlasTransposeType(bool transpose) {
  return transpose ? ep::primitive::BlasTransposeType::T : ep::primitive::BlasTransposeType::N;
}

std::unique_ptr<ep::primitive::Matmul> NewMatmulPrimitive(DeviceType device_type,
                                                          DataType data_type, bool transpose_a,
                                                          bool transpose_b) {
  const auto trans_a = GetBlasTransposeType(transpose_a);
  const auto trans_b = GetBlasTransposeType(transpose_b);
  return ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(device_type, data_type, trans_a,
                                                                   trans_b);
}

template<typename Context>
std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(Context* ctx, const int& num_dims) {
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(ctx->device_type(), num_dims);
}

template<typename T>
class DeformableConv2dCudaKernel final : public user_op::OpKernel {
 public:
  DeformableConv2dCudaKernel() = default;
  ~DeformableConv2dCudaKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* offset = ctx->Tensor4ArgNameAndIndex("offset", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ShapeView& input_shape = input->shape_view();
    const ShapeView& output_shape = output->shape_view();
    const int64_t out_elem_cnt = output_shape.elem_cnt();
    const int64_t output_bytes = GetCudaAlignedSize(out_elem_cnt * sizeof(T));

    T* column_tmp_buffer = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + output_bytes);
    const int32_t kW = weight->shape_view().At(2);
    const int32_t kH = weight->shape_view().At(3);
    const int32_t dW = ctx->Attr<int32_t>("stride_w");
    const int32_t dH = ctx->Attr<int32_t>("stride_h");
    const int32_t padW = ctx->Attr<int32_t>("pad_w");
    const int32_t padH = ctx->Attr<int32_t>("pad_h");
    const int32_t dilationW = ctx->Attr<int32_t>("dilation_w");
    const int32_t dilationH = ctx->Attr<int32_t>("dilation_h");
    const int32_t group = ctx->Attr<int32_t>("groups");
    const int32_t deformable_group = ctx->Attr<int32_t>("offset_groups");
    const bool use_mask = ctx->Attr<bool>("use_mask");

    const int32_t channel_per_deformable_group = input_shape.At(1) / deformable_group;
    const int64_t outputWidth =
        (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    const int64_t outputHeight =
        (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
    const int64_t column_nums = input_shape.At(1) * input_shape.At(0) * outputWidth * outputHeight;
    if (column_nums > 0) {
      DeformableIm2Col<T><<<BlocksNum4ThreadsNum(column_nums), kCudaThreadsNumPerBlock, 0,
                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          column_nums, input->dptr<T>(), offset->dptr<T>(), mask->dptr<T>(), input_shape.At(2),
          input_shape.At(3), kH, kW, padH, padW, dH, dW, dilationH, dilationW, input_shape.At(0),
          input_shape.At(1), deformable_group, output_shape.At(2), output_shape.At(3), use_mask,
          column_tmp_buffer);

      const int64_t weight_group_offset = weight->shape_view().elem_cnt() / group;
      const int64_t column_group_offset =
          input_shape.At(1) * kW * kH * input_shape.At(0) * outputHeight * outputWidth / group;
      const int64_t output_group_offset = out_elem_cnt / group;

      auto matmul = NewMatmulPrimitive(ctx->device_type(), output->data_type(), false, false);
      CHECK(matmul);

      FOR_RANGE(int, g, 0, group) {
        matmul->Launch(ctx->stream(), weight->shape_view().At(0) / group,
                       input_shape.At(0) * outputHeight * outputWidth,
                       input_shape.At(1) * kW * kH / group, static_cast<T>(1),
                       weight->dptr<T>() + g * weight_group_offset,
                       column_tmp_buffer + g * column_group_offset, static_cast<T>(0),
                       tmp_buffer->mut_dptr<T>() + g * output_group_offset);
      }

      std::vector<int64_t> out_shapevec(
          {output_shape.At(1), output_shape.At(0), output_shape.At(2), output_shape.At(3)});

      auto transpose = NewPermutePrimitive(ctx, output_shape.NumAxes());
      CHECK(transpose);
      transpose->Launch(ctx->stream(), output->data_type(), output_shape.NumAxes(),
                        out_shapevec.data(), tmp_buffer->dptr<T>(),
                        std::vector<int>({1, 0, 2, 3}).data(), output->mut_dptr<T>());
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
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* output_grad = ctx->Tensor4ArgNameAndIndex("output_grad", 0);
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* offset = ctx->Tensor4ArgNameAndIndex("offset", 0);
    user_op::Tensor* input_grad = ctx->Tensor4ArgNameAndIndex("input_grad", 0);
    user_op::Tensor* offset_grad = ctx->Tensor4ArgNameAndIndex("offset_grad", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ShapeView& output_grad_shape = output_grad->shape_view();
    const ShapeView& input_shape = input->shape_view();
    const ShapeView& weight_shape = weight->shape_view();
    const int32_t kW = weight->shape_view().At(2);
    const int32_t kH = weight->shape_view().At(3);
    const int32_t dW = ctx->Attr<int32_t>("stride_w");
    const int32_t dH = ctx->Attr<int32_t>("stride_h");
    const int32_t padW = ctx->Attr<int32_t>("pad_w");
    const int32_t padH = ctx->Attr<int32_t>("pad_h");
    const int32_t dilationW = ctx->Attr<int32_t>("dilation_w");
    const int32_t dilationH = ctx->Attr<int32_t>("dilation_h");
    const int32_t group = ctx->Attr<int32_t>("groups");
    const int32_t deformable_group = ctx->Attr<int32_t>("offset_groups");
    const bool use_mask = ctx->Attr<bool>("use_mask");
    const T* data_mask = nullptr;
    T* data_mask_grad = nullptr;
    if (use_mask) {
      data_mask = ctx->Tensor4ArgNameAndIndex("mask", 0)->dptr<T>();
      data_mask_grad = ctx->Tensor4ArgNameAndIndex("mask_grad", 0)->mut_dptr<T>();
    }

    const int64_t outputWidth =
        (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    const int64_t outputHeight =
        (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    std::unique_ptr<ep::primitive::Memset> primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->stream()->device_type());

    primitive->Launch(ctx->stream(), input_grad->mut_dptr<T>(), 0,
                      input_grad->shape_view().elem_cnt() * sizeof(T));

    const int64_t nthreads_coord =
        outputHeight * outputWidth * 2 * deformable_group * input_shape.At(0) * kW * kH;
    const int64_t nthreads_feat =
        outputHeight * outputWidth * input_shape.At(0) * input_shape.At(1) * kW * kH;
    if (nthreads_coord > 0 && nthreads_feat > 0) {
      const int64_t weight_group_offset = weight_shape.elem_cnt() / group;
      const int64_t output_grad_group_offset = output_grad_shape.Count(1) / group;
      const int64_t column_group_offset =
          input_shape.At(1) * kW * kH * input_shape.At(0) * outputHeight * outputWidth / group;

      auto matmul = NewMatmulPrimitive(ctx->device_type(), input_grad->data_type(), true, true);
      CHECK(matmul);
      FOR_RANGE(int, g, 0, group) {
        matmul->Launch(ctx->stream(), weight_shape.Count(1),
                       input_shape.At(0) * outputHeight * outputWidth, weight_shape.At(0) / group,
                       static_cast<T>(1), weight->dptr<T>() + g * weight_group_offset,
                       output_grad->dptr<T>() + g * output_grad_group_offset, static_cast<T>(0),
                       tmp_buffer->mut_dptr<T>() + g * column_group_offset);
      }
      DeformableCol2imCoord<T><<<BlocksNum4ThreadsNum(nthreads_coord), kCudaThreadsNumPerBlock, 0,
                                 ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          nthreads_coord, tmp_buffer->dptr<T>(), input->dptr<T>(), offset->dptr<T>(), data_mask,
          input_shape.At(1), input_shape.At(2), input_shape.At(3), kH, kW, padH, padW, dH, dW,
          dilationH, dilationW, input_shape.At(0), 2 * kH * kW * deformable_group, deformable_group,
          outputHeight, outputWidth, use_mask, offset_grad->mut_dptr<T>(), data_mask_grad);
      DeformableCol2Im<T><<<BlocksNum4ThreadsNum(nthreads_feat), kCudaThreadsNumPerBlock, 0,
                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          nthreads_feat, tmp_buffer->dptr<T>(), offset->dptr<T>(), data_mask, input_shape.At(1),
          input_shape.At(2), input_shape.At(3), kH, kW, padH, padW, dH, dW, dilationH, dilationW,
          input_shape.At(0), deformable_group, outputHeight, outputWidth, use_mask,
          input_grad->mut_dptr<T>());
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
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* output_grad = ctx->Tensor4ArgNameAndIndex("output_grad", 0);
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* offset = ctx->Tensor4ArgNameAndIndex("offset", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* weight_grad = ctx->Tensor4ArgNameAndIndex("weight_grad", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ShapeView& output_grad_shape = output_grad->shape_view();
    const ShapeView& weight_grad_shape = weight_grad->shape_view();
    const ShapeView& input_shape = input->shape_view();
    const int64_t out_elem_cnt = output_grad_shape.elem_cnt();
    const int64_t output_bytes = GetCudaAlignedSize(out_elem_cnt * sizeof(T));
    T* column_tmp_buffer = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + output_bytes);
    const int32_t kW = weight->shape_view().At(2);
    const int32_t kH = weight->shape_view().At(3);
    const int32_t dW = ctx->Attr<int32_t>("stride_w");
    const int32_t dH = ctx->Attr<int32_t>("stride_h");
    const int32_t padW = ctx->Attr<int32_t>("pad_w");
    const int32_t padH = ctx->Attr<int32_t>("pad_h");
    const int32_t dilationW = ctx->Attr<int32_t>("dilation_w");
    const int32_t dilationH = ctx->Attr<int32_t>("dilation_h");
    const int32_t group = ctx->Attr<int32_t>("groups");
    const int32_t deformable_group = ctx->Attr<int32_t>("offset_groups");
    const bool use_mask = ctx->Attr<bool>("use_mask");
    const int32_t channel_per_deformable_group = input_shape.At(1) / deformable_group;
    const int64_t outputWidth =
        (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    const int64_t outputHeight =
        (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    const T* data_mask = nullptr;
    if (use_mask) { data_mask = ctx->Tensor4ArgNameAndIndex("mask", 0)->dptr<T>(); }
    const int64_t column_nums = input_shape.At(1) * input_shape.At(0) * outputHeight * outputWidth;
    if (column_nums > 0) {
      DeformableIm2Col<T><<<BlocksNum4ThreadsNum(column_nums), kCudaThreadsNumPerBlock, 0,
                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          column_nums, input->dptr<T>(), offset->dptr<T>(), data_mask, input_shape.At(2),
          input_shape.At(3), kH, kW, padH, padW, dH, dW, dilationH, dilationW, input_shape.At(0),
          input_shape.At(1), deformable_group, output_grad_shape.At(2), output_grad_shape.At(3),
          use_mask, column_tmp_buffer);

      std::unique_ptr<ep::primitive::Memset> primitive =
          ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->stream()->device_type());
      primitive->Launch(ctx->stream(), weight_grad->mut_dptr<T>(), 0,
                        weight_grad->shape_view().elem_cnt() * sizeof(T));

      std::vector<int64_t> output_grad_buffer_vec({output_grad_shape.At(1), output_grad_shape.At(0),
                                                   output_grad_shape.At(2),
                                                   output_grad_shape.At(3)});

      auto transpose = NewPermutePrimitive(ctx, output_grad_shape.NumAxes());
      CHECK(transpose);
      transpose->Launch(ctx->stream(), output_grad->data_type(), output_grad_shape.NumAxes(),
                        output_grad_buffer_vec.data(), output_grad->dptr<T>(),
                        std::vector<int>({1, 0, 2, 3}).data(), tmp_buffer->mut_dptr<T>());

      const int64_t output_grad_group_offset = output_grad_shape.elem_cnt() / group;
      const int64_t column_group_offset =
          input_shape.At(1) * kW * kW * input_shape.At(0) * outputHeight * outputWidth / group;
      const int64_t weight_grad_group_offset = weight_grad->shape_view().elem_cnt() / group;
      FOR_RANGE(int, g, 0, group) {
        auto matmul = NewMatmulPrimitive(ctx->device_type(), weight_grad->data_type(), false, true);
        CHECK(matmul);

        matmul->Launch(ctx->stream(), weight_grad_shape.At(0) / group,
                       input_shape.At(1) * kW * kH / group,
                       input_shape.At(0) * outputHeight * outputWidth, static_cast<T>(1),
                       tmp_buffer->dptr<T>() + g * output_grad_group_offset,
                       column_tmp_buffer + g * column_group_offset, static_cast<T>(0),
                       weight_grad->mut_dptr<T>() + g * weight_grad_group_offset);
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DEFORM_CONV2D_GPU_KERNEL(dtype)                                                  \
  REGISTER_USER_KERNEL("deform_conv2d")                                                           \
      .SetCreateFn<DeformableConv2dCudaKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                            \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value))        \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const Shape& input_shape = ctx->InputShape("input", 0);                                   \
        const Shape& output_shape = ctx->OutputShape("output", 0);                                \
        const Shape& weight_shape = ctx->InputShape("weight", 0);                                 \
        const int32_t kW = weight_shape.At(2);                                                    \
        const int32_t kH = weight_shape.At(3);                                                    \
        const int32_t dW = ctx->Attr<int32_t>("stride_w");                                        \
        const int32_t dH = ctx->Attr<int32_t>("stride_h");                                        \
        const int32_t padW = ctx->Attr<int32_t>("pad_w");                                         \
        const int32_t padH = ctx->Attr<int32_t>("pad_h");                                         \
        const int32_t dilationW = ctx->Attr<int32_t>("dilation_w");                               \
        const int32_t dilationH = ctx->Attr<int32_t>("dilation_h");                               \
        const int64_t outputWidth =                                                               \
            (input_shape.At(3) + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;                 \
        const int64_t outputHeight =                                                              \
            (input_shape.At(2) + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;                 \
        const int64_t column_bytes =                                                              \
            GetCudaAlignedSize(input_shape.At(1) * kW * kH * input_shape.At(0) * outputHeight     \
                               * outputWidth * sizeof(dtype));                                    \
        const int64_t output_bytes = GetCudaAlignedSize(output_shape.elem_cnt() * sizeof(dtype)); \
        return column_bytes + output_bytes;                                                       \
      });
REGISTER_DEFORM_CONV2D_GPU_KERNEL(float)
REGISTER_DEFORM_CONV2D_GPU_KERNEL(double)

#define REGISTER_DEFORM_CONV2D_INPUT_GRAD_GPU_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("deform_conv2d_input_grad")                                            \
      .SetCreateFn<DeformableConv2dInputGradKernel<dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                        \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobDataType("weight", 0) == GetDataType<dtype>::value)    \
                       && (user_op::HobDataType("offset", 0) == GetDataType<dtype>::value))   \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                     \
        const Shape& input_shape = ctx->InputShape("input", 0);                               \
        const Shape& weight_shape = ctx->InputShape("weight", 0);                             \
        const int32_t kW = weight_shape.At(2);                                                \
        const int32_t kH = weight_shape.At(3);                                                \
        const int32_t dW = ctx->Attr<int32_t>("stride_w");                                    \
        const int32_t dH = ctx->Attr<int32_t>("stride_h");                                    \
        const int32_t padW = ctx->Attr<int32_t>("pad_w");                                     \
        const int32_t padH = ctx->Attr<int32_t>("pad_h");                                     \
        const int32_t dilationW = ctx->Attr<int32_t>("dilation_w");                           \
        const int32_t dilationH = ctx->Attr<int32_t>("dilation_h");                           \
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
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                        \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobDataType("offset", 0) == GetDataType<dtype>::value))   \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                     \
        const Shape& input_shape = ctx->InputShape("input", 0);                               \
        const Shape& output_grad_shape = ctx->InputShape("output_grad", 0);                   \
        const Shape& weight_shape = ctx->InputShape("weight", 0);                             \
        const int32_t kW = weight_shape.At(2);                                                \
        const int32_t kH = weight_shape.At(3);                                                \
        const int32_t dW = ctx->Attr<int32_t>("stride_w");                                    \
        const int32_t dH = ctx->Attr<int32_t>("stride_h");                                    \
        const int32_t padW = ctx->Attr<int32_t>("pad_w");                                     \
        const int32_t padH = ctx->Attr<int32_t>("pad_h");                                     \
        const int32_t dilationW = ctx->Attr<int32_t>("dilation_w");                           \
        const int32_t dilationH = ctx->Attr<int32_t>("dilation_h");                           \
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