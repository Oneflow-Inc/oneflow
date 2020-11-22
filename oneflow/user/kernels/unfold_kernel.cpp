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
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"
#include "oneflow/user/utils/unfold_util.h"
#include "oneflow/core/common/eigen_util.h"

namespace oneflow {

namespace {

class UnfoldCpuOpKernelState final : public user_op::OpKernelState {
 public:
  UnfoldCpuOpKernelState(ParamsUnfold3D params_3d)
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
    std::shared_ptr<UnfoldCpuOpKernelState> state(new UnfoldCpuOpKernelState(params_3d));
    return std::move(state);
  }

  ParamsUnfold3D params_3d;
};

template<typename T>
class UnfoldCpuKernelUtil {
 public:
  static void CFirstForward(const ParamsUnfold3D& params_3d, const user_op::Tensor* in_blob,
                            user_op::Tensor* out_blob) {
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
    std::memset(output, T(0), out.elem_cnt() * sizeof(T));
    FOR_RANGE(int64_t, n, 0, in.At(0)) {
      FOR_RANGE(int64_t, c, 0, in.At(1)) {
        FOR_RANGE(int64_t, pd, 0, out_5d.At(2)) {
          const int64_t dstart = pd * strides.at(0) - padding_before.at(0);
          const int64_t dend = dstart + (kernel_size.at(0) - 1) * dilation_rate.at(0) + 1;
          FOR_RANGE(int64_t, ph, 0, out_5d.At(3)) {
            const int64_t hstart = ph * strides.at(1) - padding_before.at(1);
            const int64_t hend = hstart + (kernel_size.at(1) - 1) * dilation_rate.at(1) + 1;
            FOR_RANGE(int64_t, pw, 0, out_5d.At(4)) {
              const int64_t wstart = pw * strides.at(2) - padding_before.at(2);
              const int64_t wend = wstart + (kernel_size.at(2) - 1) * dilation_rate.at(2) + 1;
              const int64_t out_col_index = pd * out_5d.Count(3) + ph * out_5d.At(4) + pw;
              int64_t out_row_index = 0;

              for (int64_t d = dstart; d < dend; d += dilation_rate.at(0)) {
                for (int64_t h = hstart; h < hend; h += dilation_rate.at(1)) {
                  for (int64_t w = wstart; w < wend; w += dilation_rate.at(2)) {
                    if (d >= 0 && h >= 0 && w >= 0 && d < in.At(2) && h < in.At(3) && w < in.At(4)) {
                      const int64_t input_index = d * in.Count(3) + h * in.At(4) + w;
                      const int64_t output_index = out_row_index * out_cols + out_col_index;
                      output[output_index] = input[input_index];
                    }
                    ++out_row_index;
                  }
                }
              }
            }
          }
        }
        input += in_channel_size;
        output += out_channel_size;
      }
    }
  }

  static void CFirstBackward(const ParamsUnfold3D& params_3d, const user_op::Tensor* out_diff_blob,
                             user_op::Tensor* in_diff_blob) {
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
    const int64_t kernel_size_extent_d = (kernel_size.at(0) - 1) * dilation_rate.at(0) + 1;
    const int64_t kernel_size_extent_h = (kernel_size.at(1) - 1) * dilation_rate.at(1) + 1;
    const int64_t kernel_size_extent_w = (kernel_size.at(2) - 1) * dilation_rate.at(2) + 1;

    const T* output_diff = out_diff_blob->dptr<T>();
    T* input_diff = in_diff_blob->mut_dptr<T>();
    std::memset(input_diff, T(0), in.elem_cnt() * sizeof(T));
    FOR_RANGE(int64_t, n, 0, in.At(0)) {
      FOR_RANGE(int64_t, c, 0, in.At(1)) {
        int64_t input_index = 0;
        FOR_RANGE(int64_t, pd_np, 0, in.At(2)) {
          const int64_t pd = pd_np + padding_before.at(0);
          const int64_t dstart = pd < kernel_size_extent_d ?
            0 : (pd - kernel_size_extent_d) / strides.at(0) + 1;
          const int64_t dend = std::min(pd / strides.at(0) + 1, out_5d.At(2));
          FOR_RANGE(int64_t, ph_np, 0, in.At(3)) {
            const int64_t ph = ph_np + padding_before.at(1);
            const int64_t hstart = ph < kernel_size_extent_h ?
              0 : (ph - kernel_size_extent_h) / strides.at(1) + 1;
            const int64_t hend = std::min(ph / strides.at(1) + 1, out_5d.At(3));
            FOR_RANGE(int64_t, pw_np, 0, in.At(4)) {
              const int64_t pw = pw_np + padding_before.at(2);
              const int64_t wstart = pw < kernel_size_extent_w ?
                0 : (pw - kernel_size_extent_w) / strides.at(2) + 1;
              const int64_t wend = std::min(pw / strides.at(2) + 1, out_5d.At(4));
              FOR_RANGE(int64_t, d, dstart, dend) {
                int64_t k_d = pd - d * strides.at(0);
                if (k_d % dilation_rate.at(0) != 0) continue;
                k_d /= dilation_rate.at(0);
                FOR_RANGE(int64_t, h, hstart, hend) {
                  int64_t k_h = ph - h * strides.at(1);
                  if (k_h % dilation_rate.at(1) != 0) continue;
                  k_h /= dilation_rate.at(1);
                  FOR_RANGE(int64_t, w, wstart, wend) {
                    int64_t k_w = pw - w * strides.at(2);
                    if (k_w % dilation_rate.at(2) != 0) continue;
                    k_w /= dilation_rate.at(2);
                    const int64_t out_row_index = (k_d * kernel_size.at(1) + k_h) * kernel_size.at(2) + k_w;
                    const int64_t out_col_index = (d * out_5d.At(1) + h) * out_5d.At(2) + w;
                    const int64_t output_index = out_row_index * out_cols + out_col_index;
                    input_diff[input_index] += output_diff[output_index];
                  }
                }
              }
              ++input_index;
            }
          }
        }
        input_diff += in_channel_size;
        output_diff += out_channel_size;
      }
    }
  }

  static void FWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto* unfold_state = dynamic_cast<UnfoldCpuOpKernelState*>(state);
    CHECK(unfold_state != nullptr);
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_first") {
      CFirstForward(unfold_state->GetParams3D(), x, y);
    } else {
      UNIMPLEMENTED();
    }
  }

  static void BWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto* unfold_state = dynamic_cast<UnfoldCpuOpKernelState*>(state);
    CHECK(unfold_state != nullptr);
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_first") {
      CFirstBackward(unfold_state->GetParams3D(), dy, dx);
    } else {
      UNIMPLEMENTED();
    }
  }
};

}  // namespace

template<typename T>
class Unfold1DCpuKernel final : public user_op::OpKernel {
 public:
  Unfold1DCpuKernel() = default;
  ~Unfold1DCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldCpuOpKernelState::DoCreateOpKernelState(ctx, 1);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldCpuKernelUtil<T>::FWCompute(ctx, state);
  };
};

template<typename T>
class Unfold1DGradCpuKernel final : public user_op::OpKernel {
 public:
  Unfold1DGradCpuKernel() = default;
  ~Unfold1DGradCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldCpuOpKernelState::DoCreateOpKernelState(ctx, 1);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldCpuKernelUtil<T>::BWCompute(ctx, state);
  };
};

template<typename T>
class Unfold2DCpuKernel final : public user_op::OpKernel {
 public:
  Unfold2DCpuKernel() = default;
  ~Unfold2DCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldCpuOpKernelState::DoCreateOpKernelState(ctx, 2);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldCpuKernelUtil<T>::FWCompute(ctx, state);
  };
};

template<typename T>
class Unfold2DGradCpuKernel final : public user_op::OpKernel {
 public:
  Unfold2DGradCpuKernel() = default;
  ~Unfold2DGradCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldCpuOpKernelState::DoCreateOpKernelState(ctx, 2);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldCpuKernelUtil<T>::BWCompute(ctx, state);
  };
};

template<typename T>
class Unfold3DCpuKernel final : public user_op::OpKernel {
 public:
  Unfold3DCpuKernel() = default;
  ~Unfold3DCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldCpuOpKernelState::DoCreateOpKernelState(ctx, 3);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldCpuKernelUtil<T>::FWCompute(ctx, state);
  };
};

template<typename T>
class Unfold3DGradCpuKernel final : public user_op::OpKernel {
 public:
  Unfold3DGradCpuKernel() = default;
  ~Unfold3DGradCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return UnfoldCpuOpKernelState::DoCreateOpKernelState(ctx, 3);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    UnfoldCpuKernelUtil<T>::BWCompute(ctx, state);
  };
};

#define REGISTER_UNFOLD_CPU_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("unfold_1d")                                                  \
      .SetCreateFn<Unfold1DCpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_1d_grad")                                             \
      .SetCreateFn<Unfold1DGradCpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_2d")                                                  \
      .SetCreateFn<Unfold2DCpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_2d_grad")                                             \
      .SetCreateFn<Unfold2DGradCpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_3d")                                                  \
      .SetCreateFn<Unfold3DCpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_3d_grad")                                             \
      .SetCreateFn<Unfold3DGradCpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_UNFOLD_CPU_KERNEL(float)
REGISTER_UNFOLD_CPU_KERNEL(double)

}  // namespace oneflow
