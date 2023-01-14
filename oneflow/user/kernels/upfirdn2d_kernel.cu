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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

struct UpFirDn2DKernelParams {
  int32_t up_x;
  int32_t up_y;
  int32_t down_x;
  int32_t down_y;
  int32_t pad_x0;
  int32_t pad_x1;
  int32_t pad_y0;
  int32_t pad_y1;

  int32_t major_dim;
  int32_t in_h;
  int32_t in_w;
  int32_t minor_dim;
  int32_t kernel_h;
  int32_t kernel_w;
  int32_t out_h;
  int32_t out_w;
  int32_t loop_major;
  int32_t loop_x;
};

struct UpFirDn2DModeParams {
  int32_t mode;
  int32_t tile_out_h;
  int32_t tile_out_w;
};

static __host__ __device__ __forceinline__ int32_t floor_div(int32_t a, int32_t b) {
  int32_t c = a / b;
  if (c * b > a) { c--; }
  return c;
}

template<typename T>
__global__ void UpFirDn2DLargeForward(const T* input, const T* kernel, T* out,
                                      const UpFirDn2DKernelParams p) {
  int32_t minor_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t out_y = minor_idx / p.minor_dim;
  minor_idx -= out_y * p.minor_dim;
  int32_t out_x_base = blockIdx.y * p.loop_x * blockDim.y + threadIdx.y;
  int32_t major_idx_base = blockIdx.z * p.loop_major;

  if (out_x_base >= p.out_w || out_y >= p.out_h || major_idx_base >= p.major_dim) { return; }

  int32_t mid_y = out_y * p.down_y + p.up_y - 1 - p.pad_y0;
  int32_t in_y = min(max(floor_div(mid_y, p.up_y), 0), p.in_h);
  int32_t h = min(max(floor_div(mid_y + p.kernel_h, p.up_y), 0), p.in_h) - in_y;
  int32_t kernel_y = mid_y + p.kernel_h - (in_y + 1) * p.up_y;

  for (int32_t loop_major = 0, major_idx = major_idx_base;
       loop_major < p.loop_major && major_idx < p.major_dim; loop_major++, major_idx++) {
    for (int32_t loop_x = 0, out_x = out_x_base; loop_x < p.loop_x && out_x < p.out_w;
         loop_x++, out_x += blockDim.y) {
      int32_t mid_x = out_x * p.down_x + p.up_x - 1 - p.pad_x0;
      int32_t in_x = min(max(floor_div(mid_x, p.up_x), 0), p.in_w);
      int32_t w = min(max(floor_div(mid_x + p.kernel_w, p.up_x), 0), p.in_w) - in_x;
      int32_t kernel_x = mid_x + p.kernel_w - (in_x + 1) * p.up_x;

      T v = 0.0f;
      const T *x_p =
          &input[((major_idx * p.in_h + in_y) * p.in_w + in_x) * p.minor_dim +
                 minor_idx];
      const T *k_p = &kernel[kernel_y * p.kernel_w + kernel_x];
      int32_t x_px = p.minor_dim;
      int32_t k_px = -p.up_x;
      int32_t x_py = p.in_w * p.minor_dim;
      int32_t k_py = -p.up_y * p.kernel_w;

      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          v += static_cast<T>(*x_p) * static_cast<T>(*k_p);
          x_p += x_px;
          k_p += k_px;
        }

        x_p += x_py - w * x_px;
        k_p += k_py - w * k_px;
      }

      out[((major_idx * p.out_h + out_y) * p.out_w + out_x) * p.minor_dim + minor_idx] =
          v;
    }
  }
}

template<typename T, int32_t up_x, int32_t up_y, int32_t down_x, int32_t down_y, int32_t kernel_h,
         int32_t kernel_w, int32_t tile_out_h, int32_t tile_out_w>
__global__ void UpFirDn2DForward(const T* input, const T* kernel, T* out,
                                 const UpFirDn2DKernelParams p) {
  const int32_t tile_in_h = ((tile_out_h - 1) * down_y + kernel_h - 1) / up_y + 1;
  const int32_t tile_in_w = ((tile_out_w - 1) * down_x + kernel_w - 1) / up_x + 1;

  __shared__ volatile float sk[kernel_h][kernel_w];
  __shared__ volatile float sx[tile_in_h][tile_in_w];

  int32_t minor_idx = blockIdx.x;
  int32_t tile_out_y = minor_idx / p.minor_dim;
  minor_idx -= tile_out_y * p.minor_dim;
  tile_out_y *= tile_out_h;
  int32_t tile_out_x_base = blockIdx.y * p.loop_x * tile_out_w;
  int32_t major_idx_base = blockIdx.z * p.loop_major;

  if (tile_out_x_base >= p.out_w | tile_out_y >= p.out_h | major_idx_base >= p.major_dim) {
    return;
  }

  for (int32_t tap_idx = threadIdx.x; tap_idx < kernel_h * kernel_w; tap_idx += blockDim.x) {
    int32_t ky = tap_idx / kernel_w;
    int32_t kx = tap_idx - ky * kernel_w;
    T v = 0.0;
    if (kx < p.kernel_w & ky < p.kernel_h) {
      v = kernel[(p.kernel_h - 1 - ky) * p.kernel_w + (p.kernel_w - 1 - kx)];
    }
    sk[ky][kx] = v;
  }

  for (int32_t loop_major = 0, major_idx = major_idx_base;
       loop_major < p.loop_major & major_idx < p.major_dim; loop_major++, major_idx++) {
    for (int32_t loop_x = 0, tile_out_x = tile_out_x_base; loop_x < p.loop_x & tile_out_x < p.out_w;
         loop_x++, tile_out_x += tile_out_w) {
      int32_t tile_mid_x = tile_out_x * down_x + up_x - 1 - p.pad_x0;
      int32_t tile_mid_y = tile_out_y * down_y + up_y - 1 - p.pad_y0;
      int32_t tile_in_x = floor_div(tile_mid_x, up_x);
      int32_t tile_in_y = floor_div(tile_mid_y, up_y);

      __syncthreads();

      for (int32_t in_idx = threadIdx.x; in_idx < tile_in_h * tile_in_w; in_idx += blockDim.x) {
        int32_t rel_in_y = in_idx / tile_in_w;
        int32_t rel_in_x = in_idx - rel_in_y * tile_in_w;
        int32_t in_x = rel_in_x + tile_in_x;
        int32_t in_y = rel_in_y + tile_in_y;

        T v = 0.0;

        if (in_x >= 0 & in_y >= 0 & in_x < p.in_w & in_y < p.in_h) {
          v = input[((major_idx * p.in_h + in_y) * p.in_w + in_x) * p.minor_dim + minor_idx];
        }

        sx[rel_in_y][rel_in_x] = v;
      }

      __syncthreads();

      for (int32_t out_idx = threadIdx.x; out_idx < tile_out_h * tile_out_w;
           out_idx += blockDim.x) {
        int32_t rel_out_y = out_idx / tile_out_w;
        int32_t rel_out_x = out_idx - rel_out_y * tile_out_w;
        int32_t out_x = rel_out_x + tile_out_x;
        int32_t out_y = rel_out_y + tile_out_y;

        int32_t mid_x = tile_mid_x + rel_out_x * down_x;
        int32_t mid_y = tile_mid_y + rel_out_y * down_y;
        int32_t in_x = floor_div(mid_x, up_x);
        int32_t in_y = floor_div(mid_y, up_y);
        int32_t rel_in_x = in_x - tile_in_x;
        int32_t rel_in_y = in_y - tile_in_y;
        int32_t kernel_x = (in_x + 1) * up_x - mid_x - 1;
        int32_t kernel_y = (in_y + 1) * up_y - mid_y - 1;

        T v = 0.0;

#pragma unroll
        for (int32_t y = 0; y < kernel_h / up_y; y++)
#pragma unroll
          for (int32_t x = 0; x < kernel_w / up_x; x++)
            v += sx[rel_in_y + y][rel_in_x + x] * sk[kernel_y + y * up_y][kernel_x + x * up_x];

        if (out_x < p.out_w & out_y < p.out_h) {
          out[((major_idx * p.out_h + out_y) * p.out_w + out_x) * p.minor_dim + minor_idx] = v;
        }
      }
    }
  }
}

UpFirDn2DKernelParams infer_upfirdn_params(
    const ShapeView input_shape, const ShapeView kernel_shape, const std::vector<int32_t>& up,
    const std::vector<int32_t>& down, const std::vector<int32_t>& pad, UpFirDn2DKernelParams p) {
  p.major_dim = input_shape.At(0);
  p.in_h = input_shape.At(1);
  p.in_w = input_shape.At(2);
  p.minor_dim = input_shape.At(3);
  p.kernel_h = kernel_shape.At(0);
  p.kernel_w = kernel_shape.At(1);
  p.up_x = up[0];
  p.up_y = up[1];
  p.down_x = down[0];
  p.down_y = down[1];
  p.pad_x0 = pad[0];
  p.pad_x1 = pad[1];
  p.pad_y0 = pad[2];
  p.pad_y1 = pad[3];
  p.out_h = (p.in_h * p.up_y + p.pad_y0 + p.pad_y1 - p.kernel_h + p.down_y) / p.down_y;
  p.out_w = (p.in_w * p.up_x + p.pad_x0 + p.pad_x1 - p.kernel_w + p.down_x) / p.down_x;
  return p;
}

UpFirDn2DModeParams infer_upfirdn_mode(UpFirDn2DKernelParams p, UpFirDn2DModeParams m) {
  m.mode = -1;
  m.tile_out_h = -1;
  m.tile_out_w = -1;

  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 1 && p.down_y == 1 && p.kernel_h <= 4
      && p.kernel_w <= 4) {
    m.mode = 1;
    m.tile_out_h = 16;
    m.tile_out_w = 64;
  }

  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 1 && p.down_y == 1 && p.kernel_h <= 3
      && p.kernel_w <= 3) {
    m.mode = 2;
    m.tile_out_h = 16;
    m.tile_out_w = 64;
  }

  if (p.up_x == 2 && p.up_y == 2 && p.down_x == 1 && p.down_y == 1 && p.kernel_h <= 4
      && p.kernel_w <= 4) {
    m.mode = 3;
    m.tile_out_h = 16;
    m.tile_out_w = 64;
  }

  if (p.up_x == 2 && p.up_y == 2 && p.down_x == 1 && p.down_y == 1 && p.kernel_h <= 2
      && p.kernel_w <= 2) {
    m.mode = 4;
    m.tile_out_h = 16;
    m.tile_out_w = 64;
  }

  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 2 && p.down_y == 2 && p.kernel_h <= 4
      && p.kernel_w <= 4) {
    m.mode = 5;
    m.tile_out_h = 8;
    m.tile_out_w = 32;
  }

  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 2 && p.down_y == 2 && p.kernel_h <= 2
      && p.kernel_w <= 2) {
    m.mode = 6;
    m.tile_out_h = 8;
    m.tile_out_w = 32;
  }
  return m;
}

template<typename T>
void DispatchUpFirDn2d(ep::Stream* stream, const T* input, const T* kernel, T* out,
                       UpFirDn2DKernelParams p, const UpFirDn2DModeParams m) {
  const int32_t mode = m.mode;
  const int32_t tile_out_h = m.tile_out_h;
  const int32_t tile_out_w = m.tile_out_w;
  auto* cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();

  dim3 block_size;
  dim3 grid_size;
  if (tile_out_h > 0 && tile_out_w > 0) {
    p.loop_major = (p.major_dim - 1) / 16384 + 1;
    p.loop_x = 1;
    block_size = dim3(32 * 8, 1, 1);
    grid_size =
        dim3(((p.out_h - 1) / tile_out_h + 1) * p.minor_dim,
             (p.out_w - 1) / (p.loop_x * tile_out_w) + 1, (p.major_dim - 1) / p.loop_major + 1);
  } else {
    p.loop_major = (p.major_dim - 1) / 16384 + 1;
    p.loop_x = 4;
    block_size = dim3(4, 32, 1);
    grid_size =
        dim3((p.out_h * p.minor_dim - 1) / block_size.x + 1,
             (p.out_w - 1) / (p.loop_x * block_size.y) + 1, (p.major_dim - 1) / p.loop_major + 1);
  }

  switch (mode) {
    case 1:
      UpFirDn2DForward<T, 1, 1, 1, 1, 4, 4, 16, 64>
          <<<grid_size, block_size, 0, cuda_stream>>>(input, kernel, out, p);
      break;
    case 2:
      UpFirDn2DForward<T, 1, 1, 1, 1, 3, 3, 16, 64>
          <<<grid_size, block_size, 0, cuda_stream>>>(input, kernel, out, p);
      break;
    case 3:
      UpFirDn2DForward<T, 2, 2, 1, 1, 4, 4, 16, 64>
          <<<grid_size, block_size, 0, cuda_stream>>>(input, kernel, out, p);
      break;
    case 4:
      UpFirDn2DForward<T, 2, 2, 1, 1, 2, 2, 16, 64>
          <<<grid_size, block_size, 0, cuda_stream>>>(input, kernel, out, p);
      break;
    case 5:
      UpFirDn2DForward<T, 1, 1, 2, 2, 4, 4, 8, 32>
          <<<grid_size, block_size, 0, cuda_stream>>>(input, kernel, out, p);
      break;
    case 6:
      UpFirDn2DForward<T, 1, 1, 2, 2, 4, 4, 8, 32>
          <<<grid_size, block_size, 0, cuda_stream>>>(input, kernel, out, p);
      break;
    default:
      UpFirDn2DLargeForward<T><<<grid_size, block_size, 0, cuda_stream>>>(input, kernel, out, p);
  }
}

}  // namespace

template<typename T>
class Upfirdn2dKernel final : public user_op::OpKernel {
 public:
  Upfirdn2dKernel() = default;
  ~Upfirdn2dKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* kernel = ctx->Tensor4ArgNameAndIndex("kernel", 0);

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const std::vector<int32_t> up = ctx->Attr<std::vector<int32_t>>("up");
    const std::vector<int32_t> down = ctx->Attr<std::vector<int32_t>>("down");
    const std::vector<int32_t> pad = ctx->Attr<std::vector<int32_t>>("pad");

    UpFirDn2DKernelParams p{};
    const ShapeView& input_shape = input->shape_view();
    const ShapeView& kernel_shape = kernel->shape_view();
    p = infer_upfirdn_params(input_shape, kernel_shape, up, down, pad, p);

    UpFirDn2DModeParams m{};
    m = infer_upfirdn_mode(p, m);

    DispatchUpFirDn2d<T>(ctx->stream(), input->dptr<T>(), kernel->dptr<T>(), out->mut_dptr<T>(), p,
                         m);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPFIRDN2D_CUDA_KERNEL(dtype)                          \
  REGISTER_USER_KERNEL("upfirdn2d")                                    \
      .SetCreateFn<Upfirdn2dKernel<dtype>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_UPFIRDN2D_CUDA_KERNEL(float)
REGISTER_UPFIRDN2D_CUDA_KERNEL(double)
// REGISTER_UPFIRDN2D_CUDA_KERNEL(half)

}  // namespace oneflow
