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
  UnfoldCpuOpKernelState(ParamsUnfold3D params_3d) : params_3d(params_3d) {}
  const ParamsUnfold3D& GetParams3D() { return params_3d; }

  static std::shared_ptr<user_op::OpKernelState> DoCreateOpKernelState(
      user_op::KernelInitContext* ctx, const int32_t& dim) {
    if (2 != dim) { UNIMPLEMENTED(); }
    const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::string& padding = ctx->Attr<std::string>("padding");
    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
    const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const std::vector<int32_t>& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
    ParamsUnfold3D params_3d =
        ParamsUnfold3D(dim, x_shape, data_format, padding, padding_before, padding_after,
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

    const T* data_im = in_blob->dptr<T>();
    T* data_col = out_blob->mut_dptr<T>();
    std::memset(data_col, T(0), out.elem_cnt() * sizeof(T));

    const int64_t channels_col =
        in.At(1) * kernel_size.at(0) * kernel_size.at(1) * kernel_size.at(2);
    for (int64_t n = 0; n < in.At(0); ++n) {
      for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
        const int64_t w_offset = c_col % kernel_size.at(2);
        const int64_t h_offset = (c_col / kernel_size.at(2)) % kernel_size.at(1);
        const int64_t d_offset =
            ((c_col / kernel_size.at(2)) / kernel_size.at(1)) % kernel_size.at(0);
        const int64_t c_im = c_col / kernel_size.at(0) / kernel_size.at(1) / kernel_size.at(2);

        for (int64_t d_col = 0; d_col < out_5d.At(2); ++d_col) {
          const int64_t d_im =
              d_col * strides.at(0) - padding_before.at(0) + d_offset * dilation_rate.at(0);

          for (int64_t h_col = 0; h_col < out_5d.At(3); ++h_col) {
            const int64_t h_im =
                h_col * strides.at(1) - padding_before.at(1) + h_offset * dilation_rate.at(1);

            for (int64_t w_col = 0; w_col < out_5d.At(4); ++w_col) {
              const int64_t w_im =
                  w_col * strides.at(2) - padding_before.at(2) + w_offset * dilation_rate.at(2);
              data_col[((c_col * out_5d.At(2) + d_col) * out_5d.At(3) + h_col) * out_5d.At(4)
                       + w_col] =
                  (d_im >= 0 && h_im >= 0 && w_im >= 0 && d_im < in.At(2) && h_im < in.At(3)
                   && w_im < in.At(4))
                      ? data_im[((c_im * in.At(2) + d_im) * in.At(3) + h_im) * in.At(4) + w_im]
                      : static_cast<T>(0);
            }
          }
        }
      }
    }
  }

  static void CFirstBackward(const ParamsUnfold3D& params_3d, const user_op::Tensor* out_diff_blob,
                             user_op::Tensor* in_diff_blob) {
    const Shape& in = params_3d.GetXShape5D();
    const Shape& out_5d = params_3d.GetYShape5D();
    const std::vector<int32_t>& kernel_size = params_3d.kernel_size_3d();
    const std::vector<int32_t>& strides = params_3d.strides_3d();
    const std::vector<int32_t>& dilation_rate = params_3d.dilation_rate_3d();
    const std::vector<int32_t>& padding_before = params_3d.padding_before_3d();

    const T* data_col = out_diff_blob->dptr<T>();
    T* data_im = in_diff_blob->mut_dptr<T>();
    std::memset(data_im, T(0), in.elem_cnt() * sizeof(T));

    const int64_t channels_col =
        in.At(1) * kernel_size.at(0) * kernel_size.at(1) * kernel_size.at(2);
    for (int64_t n = 0; n < in.At(0); ++n) {
      for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
        const int64_t d_offset = c_col % kernel_size.at(0);
        const int64_t h_offset = (c_col / kernel_size.at(0)) % kernel_size.at(1);
        const int64_t w_offset =
            ((c_col / kernel_size.at(0)) / kernel_size.at(1)) % kernel_size.at(2);
        const int64_t c_im = c_col / kernel_size.at(0) / kernel_size.at(1) / kernel_size.at(2);

        for (int64_t d_col = 0; d_col < out_5d.At(2); ++d_col) {
          const int64_t d_im =
              d_col * strides.at(0) - padding_before.at(0) + d_offset * dilation_rate.at(0);

          for (int64_t h_col = 0; h_col < out_5d.At(3); ++h_col) {
            const int64_t h_im =
                h_col * strides.at(1) - padding_before.at(1) + h_offset * dilation_rate.at(1);

            for (int64_t w_col = 0; w_col < out_5d.At(4); ++w_col) {
              const int64_t w_im =
                  w_col * strides.at(2) - padding_before.at(2) + w_offset * dilation_rate.at(2);

              if (d_im >= 0 && h_im >= 0 && w_im >= 0 && d_im < in.At(2) && h_im < in.At(3)
                  && w_im < in.At(4)) {
                data_im[((c_im * in.At(2) + d_im) * in.At(3) + h_im) * in.At(4) + w_im] +=
                    data_col[((c_col * out_5d.At(2) + d_col) * out_5d.At(3) + h_col) * out_5d.At(4)
                             + w_col];
              }
            }
          }
        }
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

#define REGISTER_UNFOLD_CPU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("unfold_1d")                                                    \
      .SetCreateFn<Unfold1DCpuKernel<dtype>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_1d_grad")                                               \
      .SetCreateFn<Unfold1DGradCpuKernel<dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_2d")                                                    \
      .SetCreateFn<Unfold2DCpuKernel<dtype>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_2d_grad")                                               \
      .SetCreateFn<Unfold2DGradCpuKernel<dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_3d")                                                    \
      .SetCreateFn<Unfold3DCpuKernel<dtype>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("unfold_3d_grad")                                               \
      .SetCreateFn<Unfold3DGradCpuKernel<dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_UNFOLD_CPU_KERNEL(float)
REGISTER_UNFOLD_CPU_KERNEL(double)

}  // namespace oneflow
