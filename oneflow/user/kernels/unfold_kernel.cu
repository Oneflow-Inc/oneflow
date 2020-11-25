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
                                    const int64_t in_d, const int64_t in_h,
                                    const int64_t in_w, 
                                    const int64_t out_5d_d, const int64_t out_5d_h,
                                    const int64_t out_5d_w,
                                    const int64_t out_cols,
                                    const int64_t strides_d, const int64_t strides_h,
                                    const int64_t strides_w,
                                    const int64_t padding_before_d, const int64_t padding_before_h,
                                    const int64_t padding_before_w,
                                    const int64_t kernel_size_d, const int64_t kernel_size_h,
                                    const int64_t kernel_size_w,
                                    const int64_t dilation_rate_d, const int64_t dilation_rate_h,
                                    const int64_t dilation_rate_w, 
                                    const int64_t in_channel_size, const int64_t out_channel_size,
                                    const T* input, T* output) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t idx = index;
    const int64_t pw = idx % out_5d_w;
    idx /= out_5d_w;
    const int64_t ph = idx % out_5d_h;
    idx /= out_5d_h;
    const int64_t pd = idx % out_5d_d;
    idx /= out_5d_d;
    input += in_channel_size * idx;
    output += out_channel_size * idx;

    const int64_t dstart = pd * strides_d - padding_before_d;
    const int64_t dend = dstart + (kernel_size_d - 1) * dilation_rate_d + 1;
    const int64_t hstart = ph * strides_h - padding_before_h;
    const int64_t hend = hstart + (kernel_size_h - 1) * dilation_rate_h + 1;
    const int64_t wstart = pw * strides_w - padding_before_w;
    const int64_t wend = wstart + (kernel_size_w - 1) * dilation_rate_w + 1;
    const int64_t out_col_index = (pd * out_5d_h + ph) * out_5d_w + pw;
    int64_t out_row_index = 0;

    for (int64_t d = dstart; d < dend; d += dilation_rate_d) {
      for (int64_t h = hstart; h < hend; h += dilation_rate_h) {
        for (int64_t w = wstart; w < wend; w += dilation_rate_w) {
          if (d >= 0 && h >= 0 && w >= 0 && d < in_d && h < in_h && w < in_w) {
            const int64_t input_index = (d * in_h + h) * in_w + w;
            const int64_t output_index = out_row_index * out_cols + out_col_index;
            output[output_index] = input[input_index];
          }
          ++out_row_index;
        }
      }
    }
  }
}

template<typename T>
__global__ void UnfoldCFirstBackward(const int64_t elem_cnt, 
                                     const int64_t in_d, const int64_t in_h,
                                     const int64_t in_w, 
                                     const int64_t out_5d_d, const int64_t out_5d_h,
                                     const int64_t out_5d_w,
                                     const int64_t out_cols,
                                     const int64_t strides_d, const int64_t strides_h,
                                     const int64_t strides_w,
                                     const int64_t padding_before_d, const int64_t padding_before_h,
                                     const int64_t padding_before_w,
                                     const int64_t kernel_size_d, const int64_t kernel_size_h,
                                     const int64_t kernel_size_w,
                                     const int64_t dilation_rate_d, const int64_t dilation_rate_h,
                                     const int64_t dilation_rate_w, 
                                     const int64_t in_channel_size, const int64_t out_channel_size,
                                     const T* output_diff, T* input_diff) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t idx = index;
    const int64_t pw = idx % in_w + padding_before_w;
    idx /= in_w;
    const int64_t ph = idx % in_h + padding_before_h;
    idx /= in_h;
    const int64_t pd = idx % in_d + padding_before_d;
    idx /= in_d;
    output_diff += out_channel_size * idx;

    const int64_t kernel_size_extent_d = (kernel_size_d - 1) * dilation_rate_d + 1;
    const int64_t kernel_size_extent_h = (kernel_size_h - 1) * dilation_rate_h + 1;
    const int64_t kernel_size_extent_w = (kernel_size_w - 1) * dilation_rate_w + 1;
    const int64_t dstart = pd < kernel_size_extent_d ? 0 : (pd - kernel_size_extent_d) / strides_d + 1;
    const int64_t dend = min(pd / strides_d + 1, out_5d_d);
    const int64_t hstart = ph < kernel_size_extent_h ? 0 : (ph - kernel_size_extent_h) / strides_h + 1;
    const int64_t hend = min(ph / strides_h + 1, out_5d_h);
    const int64_t wstart = pw < kernel_size_extent_w ? 0 : (pw - kernel_size_extent_w) / strides_w + 1;
    const int64_t wend = min(pw / strides_w + 1, out_5d_w);

    T val = static_cast<T>(0);
    for (int64_t d = dstart; d < dend; d += 1) {
      for (int64_t h = hstart; h < hend; h += 1) {
        for (int64_t w = wstart; w < wend; w += 1) {
          int64_t k_d = (pd - d * strides_d);
          int64_t k_h = (ph - h * strides_h);
          int64_t k_w = (pw - w * strides_w);
          if (k_d % dilation_rate_d == 0 && k_h % dilation_rate_h == 0 && k_w % dilation_rate_w == 0) {
            k_d /= dilation_rate_d;
            k_h /= dilation_rate_h;
            k_w /= dilation_rate_w;
            const int64_t out_row_index = (k_d * kernel_size_h + k_h) * kernel_size_w + k_w;
            const int64_t out_col_index = (d * out_5d_h + h) * out_5d_w + w;
            const int64_t output_index = out_row_index * out_cols + out_col_index;
            val += output_diff[output_index];
          }
        }
      }
    }
    input_diff[index] = val;
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
    const int64_t out_cols = out.At(2);
    const int64_t in_channel_size = in.Count(2);
    const int64_t out_channel_size = out.At(1) / in.At(1) * out_cols;

    const T* input = in_blob->dptr<T>();
    T* output = out_blob->mut_dptr<T>();
    cudaMemset(output, T(0), out.elem_cnt() * sizeof(T));

    UnfoldCFirstForward<T><<<GetUnfoldMaxNumBlocks(out_5d.elem_cnt()),
                              kUnfoldMaxGpuBlockSize, 0, device_ctx->cuda_stream()>>>(
      out_5d.elem_cnt(),
      in.At(2), in.At(3), in.At(4),
      out_5d.At(2), out_5d.At(3), out_5d.At(4), out_cols,
      strides.at(0), strides.at(1), strides.at(2),
      padding_before.at(0), padding_before.at(1), padding_before.at(2),
      kernel_size.at(0), kernel_size.at(1), kernel_size.at(2),
      dilation_rate.at(0), dilation_rate.at(1), dilation_rate.at(2),
      in_channel_size, out_channel_size, input, output);
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

    const T* output_diff = out_diff_blob->dptr<T>();
    T* input_diff = in_diff_blob->mut_dptr<T>();
    cudaMemset(input_diff, T(0), in.elem_cnt() * sizeof(T));

    UnfoldCFirstBackward<T><<<GetUnfoldMaxNumBlocks(in.elem_cnt()),
                              kUnfoldMaxGpuBlockSize, 0, device_ctx->cuda_stream()>>>(
      in.elem_cnt(),
      in.At(2), in.At(3), in.At(4),
      out_5d.At(2), out_5d.At(3), out_5d.At(4), out_cols,
      strides.at(0), strides.at(1), strides.at(2),
      padding_before.at(0), padding_before.at(1), padding_before.at(2),
      kernel_size.at(0), kernel_size.at(1), kernel_size.at(2),
      dilation_rate.at(0), dilation_rate.at(1), dilation_rate.at(2),
      in_channel_size, out_channel_size, output_diff, input_diff);
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
