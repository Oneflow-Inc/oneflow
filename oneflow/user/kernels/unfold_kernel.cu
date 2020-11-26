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
#ifdef WITH_CUDA

#include "oneflow/core/framework/framework.h"
#include "oneflow/user/utils/unfold_util.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

namespace {

constexpr int32_t kUnfoldMaxGpuBlockSize = 256;

template<typename T>
__global__ void UnfoldCFirstForward(const int64_t elem_cnt,
                                    const T* data_im,
                                    const int64_t depth,
                                    const int64_t height,
                                    const int64_t width,
                                    const int64_t kernel_depth,
                                    const int64_t kernel_height,
                                    const int64_t kernel_width,
                                    const int64_t pad_depth,
                                    const int64_t pad_height,
                                    const int64_t pad_width,
                                    const int64_t stride_depth,
                                    const int64_t stride_height,
                                    const int64_t stride_width,
                                    const int64_t dilation_depth,
                                    const int64_t dilation_height,
                                    const int64_t dilation_width,
                                    const int64_t depth_col,
                                    const int64_t height_col,
                                    const int64_t width_col,
                                    T* data_col) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t w_out = index % width_col;
    int64_t idx = index / width_col;
    int64_t h_out = idx % height_col;
    idx /= height_col;
    int64_t d_out = idx % depth_col;

    int64_t channel_in = idx / depth_col;
    int64_t channel_out = channel_in * kernel_depth * kernel_height * kernel_width;
    int64_t d_in = d_out * stride_depth - pad_depth;
    int64_t h_in = h_out * stride_height - pad_height;
    int64_t w_in = w_out * stride_width - pad_width;

    T* col = data_col + ((channel_out * depth_col + d_out) * height_col + h_out) * width_col + w_out;
    const T* im = data_im + ((channel_in * depth + d_in ) * height + h_in) * width + w_in;

    for (int64_t i = 0; i < kernel_depth; ++i) {
      for (int64_t j = 0; j < kernel_height; ++j) {
        for (int64_t k = 0; k < kernel_width; ++k) {
          int64_t d = d_in + i * dilation_depth;
          int64_t h = h_in + j * dilation_height;
          int64_t w = w_in + k * dilation_width;
          *col = (d >= 0 && h >= 0 && w >= 0 && d < depth && h < height && w < width)
              ? im[(i * dilation_depth * height + j * dilation_height) * width + k * dilation_width]
              : static_cast<T>(0);
          col += depth_col * height_col * width_col;
        }
      }
    }
  }
}

template<typename T>
__global__ void UnfoldCFirstBackward(const int64_t elem_cnt,
                                     const T* data_col,
                                     const int64_t depth,
                                     const int64_t height,
                                     const int64_t width,
                                     const int64_t kernel_d,
                                     const int64_t kernel_h,
                                     const int64_t kernel_w,
                                     const int64_t pad_depth,
                                     const int64_t pad_height,
                                     const int64_t pad_width,
                                     const int64_t stride_depth,
                                     const int64_t stride_height,
                                     const int64_t stride_width,
                                     const int64_t dilation_depth,
                                     const int64_t dilation_height,
                                     const int64_t dilation_width,
                                     const int64_t depth_col,
                                     const int64_t height_col,
                                     const int64_t width_col,
                                     T* data_im) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    T val = static_cast<T>(0);
    const int64_t w_im = index % width + pad_width;
    const int64_t h_im = (index / width) % height + pad_height;
    const int64_t d_im = ((index / width) / height) % depth + pad_depth;
    const int64_t c_im = index / (width * height * depth);
    int64_t kernel_extent_w = (kernel_w - 1) * dilation_width + 1;
    int64_t kernel_extent_h = (kernel_h - 1) * dilation_height + 1;
    int64_t kernel_extent_d = (kernel_d - 1) * dilation_depth + 1;

    // compute the start and end of the output
    const int64_t w_col_start = (w_im < kernel_extent_w)
        ? 0
        : (w_im - kernel_extent_w) / stride_width + 1;
    const int64_t w_col_end = ::min(w_im / stride_width + 1, width_col);
    const int64_t h_col_start = (h_im < kernel_extent_h)
        ? 0
        : (h_im - kernel_extent_h) / stride_height + 1;
    const int64_t h_col_end = ::min(h_im / stride_height + 1, height_col);
    const int64_t d_col_start = (d_im < kernel_extent_d)
        ? 0
        : (d_im - kernel_extent_d) / stride_depth + 1;
    const int64_t d_col_end = ::min(d_im / stride_depth + 1, depth_col);

    for (int64_t d_col = d_col_start; d_col < d_col_end; d_col += 1) {
      for (int64_t h_col = h_col_start; h_col < h_col_end; h_col += 1) {
        for (int64_t w_col = w_col_start; w_col < w_col_end; w_col += 1) {
          int64_t d_k = (d_im - d_col * stride_depth);
          int64_t h_k = (h_im - h_col * stride_height);
          int64_t w_k = (w_im - w_col * stride_width);
          if (d_k % dilation_depth == 0 && h_k % dilation_height == 0 && w_k % dilation_width == 0) {
            d_k /= dilation_depth;
            h_k /= dilation_height;
            w_k /= dilation_width;
            int64_t data_col_index =
                (((((c_im * kernel_d + d_k) * kernel_h + h_k) * kernel_w + w_k) * depth_col + d_col) * height_col +
                h_col) *
                    width_col +
                w_col;
            val += data_col[data_col_index];
          }
        }
      }
    }
    data_im[index] = static_cast<T>(val);
  }
}

inline int32_t GetUnfoldMaxNumBlocks(const int32_t n) {
  CHECK_GT(n, 0);
  return std::min((n + kUnfoldMaxGpuBlockSize - 1) / kUnfoldMaxGpuBlockSize, kCudaMaxBlocksNum);
}

class UnfoldGpuOpKernelState final : public user_op::OpKernelState {
 public:
  UnfoldGpuOpKernelState(ParamsUnfold3D params_3d)
      : params_3d(params_3d) {}
  const ParamsUnfold3D& GetParams3D() { return params_3d; }

  static std::shared_ptr<user_op::OpKernelState> DoCreateOpKernelState(
      user_op::KernelInitContext* ctx, const int32_t& dim) {
    if (2 != dim) {
        UNIMPLEMENTED();
    }
    const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::string& padding = ctx->Attr<std::string>("padding");
    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
    const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const std::vector<int32_t>& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
    ParamsUnfold3D params_3d = ParamsUnfold3D(dim, x_shape, data_format, padding, padding_before, padding_after,
                                  kernel_size, strides, dilation_rate, ceil_mode);
    std::shared_ptr<UnfoldGpuOpKernelState> state(new UnfoldGpuOpKernelState(params_3d));
    return std::move(state);
  }

  ParamsUnfold3D params_3d;
};

template<typename T>
struct UnfoldGpuKernelUtil {
 public:
  static void CFirstForward(const DeviceCtx* device_ctx, const ParamsUnfold3D& params_3d,
                            const user_op::Tensor* in_blob, user_op::Tensor* out_blob) {
    const Shape& in = params_3d.GetXShape5D();
    const Shape& out_5d = params_3d.GetYShape5D();
    const Shape& out = params_3d.GetYShape();
    const std::vector<int32_t>& kernel_size = params_3d.kernel_size_3d();
    const std::vector<int32_t>& strides = params_3d.strides_3d();
    const std::vector<int32_t>& dilation_rate = params_3d.dilation_rate_3d();
    const std::vector<int32_t>& padding_before = params_3d.padding_before_3d();

    const T* data_im = in_blob->dptr<T>();
    T* data_col = out_blob->mut_dptr<T>();
    cudaMemset(data_col, T(0), out.elem_cnt() * sizeof(T));

    UnfoldCFirstForward<T><<<GetUnfoldMaxNumBlocks(out_5d.elem_cnt()),
                              kUnfoldMaxGpuBlockSize, 0, device_ctx->cuda_stream()>>>(
      out_5d.elem_cnt(), data_im,
      in.At(2), in.At(3), in.At(4),
      kernel_size.at(0), kernel_size.at(1), kernel_size.at(2),
      padding_before.at(0), padding_before.at(1), padding_before.at(2),
      strides.at(0), strides.at(1), strides.at(2),
      dilation_rate.at(0), dilation_rate.at(1), dilation_rate.at(2),
      out_5d.At(2), out_5d.At(3), out_5d.At(4),
      data_col);
    OF_CUDA_CHECK(cudaGetLastError());

  }

  static void CFirstBackward(const DeviceCtx* device_ctx, const ParamsUnfold3D& params_3d,
                             const user_op::Tensor* out_diff_blob, user_op::Tensor* in_diff_blob) {
    const Shape& in = params_3d.GetXShape5D();
    const Shape& out_5d = params_3d.GetYShape5D();
    const Shape& out = params_3d.GetYShape();
    const std::vector<int32_t>& kernel_size = params_3d.kernel_size_3d();
    const std::vector<int32_t>& strides = params_3d.strides_3d();
    const std::vector<int32_t>& dilation_rate = params_3d.dilation_rate_3d();
    const std::vector<int32_t>& padding_before = params_3d.padding_before_3d();
    const int64_t out_cols = out.At(2);
    const int64_t in_channel_size = in.Count(2);
    const int64_t out_channel_size = out.At(1) / in.At(1) * out_cols;

    const T* data_col = out_diff_blob->dptr<T>();
    T* data_im = in_diff_blob->mut_dptr<T>();
    cudaMemset(data_im, T(0), in.elem_cnt() * sizeof(T));

    UnfoldCFirstBackward<T><<<GetUnfoldMaxNumBlocks(in.elem_cnt()),
                              kUnfoldMaxGpuBlockSize, 0, device_ctx->cuda_stream()>>>(
      in.elem_cnt(), data_col,
      in.At(2), in.At(3), in.At(4),
      kernel_size.at(0), kernel_size.at(1), kernel_size.at(2),
      padding_before.at(0), padding_before.at(1), padding_before.at(2),
      strides.at(0), strides.at(1), strides.at(2),
      dilation_rate.at(0), dilation_rate.at(1), dilation_rate.at(2),
      out_5d.At(2), out_5d.At(3), out_5d.At(4),
      data_im);
    OF_CUDA_CHECK(cudaGetLastError());

  }

  static void FWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto* unfold_state = dynamic_cast<UnfoldGpuOpKernelState*>(state);
    CHECK(unfold_state != nullptr);
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_first") {
      CFirstForward(ctx->device_ctx(), unfold_state->GetParams3D(), x, y);
    } else {
      UNIMPLEMENTED();
    }
  }

  static void BWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto* unfold_state = dynamic_cast<UnfoldGpuOpKernelState*>(state);
    CHECK(unfold_state != nullptr);
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_first") {
      CFirstBackward(ctx->device_ctx(), unfold_state->GetParams3D(), dy, dx);
    } else {
      UNIMPLEMENTED();
    }
  }
};

}  // namespace

template<typename T>
class Unfold1DGpuKernel final : public user_op::OpKernel {
 public:
  Unfold1DGpuKernel() = default;
  ~Unfold1DGpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldGpuOpKernelState::DoCreateOpKernelState(ctx, 1);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldGpuKernelUtil<T>::FWCompute(ctx, state);
  };
};

template<typename T>
class Unfold1DGradGpuKernel final : public user_op::OpKernel {
 public:
  Unfold1DGradGpuKernel() = default;
  ~Unfold1DGradGpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldGpuOpKernelState::DoCreateOpKernelState(ctx, 1);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldGpuKernelUtil<T>::BWCompute(ctx, state);
  };
};

template<typename T>
class Unfold2DGpuKernel final : public user_op::OpKernel {
 public:
  Unfold2DGpuKernel() = default;
  ~Unfold2DGpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldGpuOpKernelState::DoCreateOpKernelState(ctx, 2);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldGpuKernelUtil<T>::FWCompute(ctx, state);
  };
};

template<typename T>
class Unfold2DGradGpuKernel final : public user_op::OpKernel {
 public:
  Unfold2DGradGpuKernel() = default;
  ~Unfold2DGradGpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldGpuOpKernelState::DoCreateOpKernelState(ctx, 2);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldGpuKernelUtil<T>::BWCompute(ctx, state);
  };
};

template<typename T>
class Unfold3DGpuKernel final : public user_op::OpKernel {
 public:
  Unfold3DGpuKernel() = default;
  ~Unfold3DGpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldGpuOpKernelState::DoCreateOpKernelState(ctx, 3);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldGpuKernelUtil<T>::FWCompute(ctx, state);
  };
};

template<typename T>
class Unfold3DGradGpuKernel final : public user_op::OpKernel {
 public:
  Unfold3DGradGpuKernel() = default;
  ~Unfold3DGradGpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldGpuOpKernelState::DoCreateOpKernelState(ctx, 3);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldGpuKernelUtil<T>::BWCompute(ctx, state);
  };
};

#define REGISTER_UNFOLD_GPU_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("unfold_1d")                                                  \
      .SetCreateFn<Unfold1DGpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_1d_grad")                                             \
      .SetCreateFn<Unfold1DGradGpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_2d")                                                  \
      .SetCreateFn<Unfold2DGpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_2d_grad")                                             \
      .SetCreateFn<Unfold2DGradGpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_3d")                                                  \
      .SetCreateFn<Unfold3DGpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_3d_grad")                                             \
      .SetCreateFn<Unfold3DGradGpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_UNFOLD_GPU_KERNEL(float)
REGISTER_UNFOLD_GPU_KERNEL(double)

}  // namespace oneflow

#endif
