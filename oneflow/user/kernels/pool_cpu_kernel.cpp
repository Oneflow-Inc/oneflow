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
#include "oneflow/user/utils/pool_util.h"
#include "oneflow/core/common/eigen_util.h"

namespace oneflow {

namespace {

struct PoolOpKernelState final : public user_op::OpKernelState {
  Params3D params_3d;
  bool is_dynamic;
  PoolOpKernelState(Params3D params_3d, bool is_dynamic)
      : params_3d(params_3d), is_dynamic(is_dynamic) {}
  const Params3D& GetParams3D() { return params_3d; }
  void Update(const ShapeView& x_shape) {
    if (is_dynamic) { params_3d.Reset(x_shape); }
  }
};

template<typename T>
struct PoolCpuKernelUtil {
 public:
  typedef std::function<T()> ForwardInitialize;
  typedef std::function<void(const T& lhs, T& rhs)> CFirstProcess;
  typedef std::function<void(const int64_t in_col, const int64_t out_col,
                             ConstEigenMatrixMap<T>& in_mat, EigenMatrixMap<T>& out_mat)>
      CLastProcess;
  typedef std::function<void(const int64_t size, T& out)> CFirstFinalize;
  typedef std::function<void(const int64_t size, const int64_t col, EigenMatrixMap<T>& out_mat)>
      CLastFinalize;
  typedef std::function<void(const T& in, const T& out, const T& out_diff, const int64_t size,
                             T& in_diff)>
      CFirstProcessGrad;
  typedef std::function<void(const int64_t out_col, const int64_t in_col, const int64_t size,
                             ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
                             ConstEigenArrayMap<T>& out_diff_arr, EigenArrayMap<T>& in_diff_arr)>
      CLastProcessGrad;

  static void CFirstForward(const Params3D& params_3d, const user_op::Tensor* in_blob,
                            user_op::Tensor* out_blob, const ForwardInitialize& initialize,
                            const CFirstProcess& process, const CFirstFinalize& finalize) {
    const Shape& in = params_3d.GetXShape5D();
    const Shape& out = params_3d.GetYShape5D();
    const std::vector<int32_t>& pool_size = params_3d.pool_size_3d();
    const std::vector<int32_t>& strides = params_3d.strides_3d();
    const std::vector<int32_t>& padding_before = params_3d.padding_before_3d();

    const T* input = in_blob->dptr<T>();
    T* output = out_blob->mut_dptr<T>();
    FOR_RANGE(int64_t, n, 0, in.At(0)) {
      FOR_RANGE(int64_t, c, 0, in.At(1)) {
        FOR_RANGE(int64_t, pd, 0, out.At(2)) {
          int64_t dstart = pd * strides.at(0) - padding_before.at(0);
          int64_t dend = std::min(dstart + pool_size.at(0), in.At(2));
          dstart = std::max(dstart, static_cast<int64_t>(0));
          FOR_RANGE(int64_t, ph, 0, out.At(3)) {
            int64_t hstart = ph * strides.at(1) - padding_before.at(1);
            int64_t hend = std::min(hstart + pool_size.at(1), in.At(3));
            hstart = std::max(hstart, static_cast<int64_t>(0));
            FOR_RANGE(int64_t, pw, 0, out.At(4)) {
              int64_t wstart = pw * strides.at(2) - padding_before.at(2);
              int64_t wend = std::min(wstart + pool_size.at(2), in.At(4));
              wstart = std::max(wstart, static_cast<int64_t>(0));

              const int64_t pool_index = pd * out.Count(3) + ph * out.At(4) + pw;
              T res = initialize();
              FOR_RANGE(int64_t, d, dstart, dend) {
                FOR_RANGE(int64_t, h, hstart, hend) {
                  FOR_RANGE(int64_t, w, wstart, wend) {
                    const int64_t input_index = d * in.Count(3) + h * in.At(4) + w;
                    process(input[input_index], res);
                  }
                }
              }
              finalize((dend - dstart) * (hend - hstart) * (wend - wstart), res);
              output[pool_index] = res;
            }
          }
        }
        input += in.Count(2);
        output += out.Count(2);
      }
    }
  }

  static void CFirstBackward(const Params3D& params_3d, const user_op::Tensor* out_diff_blob,
                             const user_op::Tensor* out_blob, const user_op::Tensor* in_blob,
                             user_op::Tensor* in_diff_blob, const CFirstProcessGrad& process) {
    const Shape& in = params_3d.GetXShape5D();
    const Shape& out = params_3d.GetYShape5D();
    const std::vector<int32_t>& pool_size = params_3d.pool_size_3d();
    const std::vector<int32_t>& strides = params_3d.strides_3d();
    const std::vector<int32_t>& padding_before = params_3d.padding_before_3d();

    const T* output_diff = out_diff_blob->dptr<T>();
    const T* output = out_blob->dptr<T>();
    const T* input = in_blob->dptr<T>();
    std::memset(in_diff_blob->mut_dptr<T>(), T(0), in.elem_cnt() * sizeof(T));
    T* input_diff = in_diff_blob->mut_dptr<T>();
    FOR_RANGE(int64_t, n, 0, in.At(0)) {
      FOR_RANGE(int64_t, c, 0, in.At(1)) {
        FOR_RANGE(int64_t, pd, 0, out.At(2)) {
          int64_t dstart = pd * strides.at(0) - padding_before.at(0);
          int64_t dend = std::min(dstart + pool_size.at(0), in.At(2));
          dstart = std::max(dstart, static_cast<int64_t>(0));
          FOR_RANGE(int64_t, ph, 0, out.At(3)) {
            int64_t hstart = ph * strides.at(1) - padding_before.at(1);
            int64_t hend = std::min(hstart + pool_size.at(1), in.At(3));
            hstart = std::max(hstart, static_cast<int64_t>(0));
            FOR_RANGE(int64_t, pw, 0, out.At(4)) {
              int64_t wstart = pw * strides.at(2) - padding_before.at(2);
              int64_t wend = std::min(wstart + pool_size.at(2), in.At(4));
              wstart = std::max(wstart, static_cast<int64_t>(0));

              const int64_t size = (dend - dstart) * (hend - hstart) * (wend - wstart);
              const int64_t pool_index = pd * out.Count(3) + ph * out.At(4) + pw;
              FOR_RANGE(int64_t, d, dstart, dend) {
                FOR_RANGE(int64_t, h, hstart, hend) {
                  FOR_RANGE(int64_t, w, wstart, wend) {
                    const int64_t index = d * in.Count(3) + h * in.At(4) + w;
                    process(input[index], output[pool_index], output_diff[pool_index], size,
                            input_diff[index]);
                  }
                }
              }
            }
          }
        }
        // offset
        input += in.Count(2);
        input_diff += in.Count(2);
        output += out.Count(2);
        output_diff += out.Count(2);
      }
    }
  }

  static void CLastForward(const Params3D& params_3d, const user_op::Tensor* in_blob,
                           user_op::Tensor* out_blob, const ForwardInitialize& forward_initialize,
                           const CLastProcess& process, const CLastFinalize& finalize) {
    const Shape& in = params_3d.GetXShape5D();
    const Shape& out = params_3d.GetYShape5D();
    const std::vector<int32_t>& pool_size = params_3d.pool_size_3d();
    const std::vector<int32_t>& strides = params_3d.strides_3d();
    const std::vector<int32_t>& padding_before = params_3d.padding_before_3d();

    ConstEigenMatrixMap<T> in_mat(in_blob->dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
    EigenMatrixMap<T> out_mat(out_blob->mut_dptr<T>(), out.At(1), out.elem_cnt() / out.At(1));
    FOR_RANGE(int64_t, n, 0, in.At(0)) {
      FOR_RANGE(int64_t, pd, 0, out.At(2)) {
        int64_t dstart = pd * strides.at(0) - padding_before.at(0);
        int64_t dend = std::min(dstart + pool_size.at(0), in.At(2));
        dstart = std::max(dstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, ph, 0, out.At(3)) {
          int64_t hstart = ph * strides.at(1) - padding_before.at(1);
          int64_t hend = std::min(hstart + pool_size.at(1), in.At(3));
          hstart = std::max(hstart, static_cast<int64_t>(0));
          FOR_RANGE(int64_t, pw, 0, out.At(4)) {
            int64_t wstart = pw * strides.at(2) - padding_before.at(2);
            int64_t wend = std::min(wstart + pool_size.at(2), in.At(4));
            wstart = std::max(wstart, static_cast<int64_t>(0));
            const int out_col = ((n * out.At(2) + pd) * out.At(3) + ph) * out.At(4) + pw;
            out_mat.col(out_col).setConstant(forward_initialize());
            FOR_RANGE(int64_t, d, dstart, dend) {
              FOR_RANGE(int64_t, h, hstart, hend) {
                FOR_RANGE(int64_t, w, wstart, wend) {
                  const int in_col = ((n * in.At(2) + d) * in.At(3) + h) * in.At(4) + w;
                  process(in_col, out_col, in_mat, out_mat);
                }
              }
            }
            finalize((hend - hstart) * (wend - wstart) * (dend - dstart), out_col, out_mat);
          }
        }
      }
    }
  }

  static void CLastBackward(const Params3D& params_3d, const user_op::Tensor* out_diff_blob,
                            const user_op::Tensor* out_blob, const user_op::Tensor* in_blob,
                            user_op::Tensor* in_diff_blob, const CLastProcessGrad& process) {
    const Shape& in = params_3d.GetXShape5D();
    const Shape& out = params_3d.GetYShape5D();
    const std::vector<int32_t>& pool_size = params_3d.pool_size_3d();
    const std::vector<int32_t>& strides = params_3d.strides_3d();
    const std::vector<int32_t>& padding_before = params_3d.padding_before_3d();

    // caffe2 implementation: need check
    ConstEigenArrayMap<T> out_mat(out_blob->dptr<T>(), out.At(1), out.elem_cnt() / out.At(1));
    ConstEigenArrayMap<T> in_mat(in_blob->dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
    ConstEigenArrayMap<T> out_diff_mat(out_diff_blob->dptr<T>(), out.At(1),
                                       out.elem_cnt() / out.At(1));
    std::memset(in_diff_blob->mut_dptr<T>(), T(0), in.elem_cnt() * sizeof(T));
    EigenArrayMap<T> in_diff_mat(in_diff_blob->mut_dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
    FOR_RANGE(int64_t, n, 0, in.At(0)) {
      FOR_RANGE(int64_t, pd, 0, out.At(2)) {
        int64_t dstart = pd * strides.at(0) - padding_before.at(0);
        int64_t dend = std::min(dstart + pool_size.at(0), in.At(2));
        dstart = std::max(dstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, ph, 0, out.At(3)) {
          int64_t hstart = ph * strides.at(1) - padding_before.at(1);
          int64_t hend = std::min(hstart + pool_size.at(1), in.At(3));
          hstart = std::max(hstart, static_cast<int64_t>(0));
          FOR_RANGE(int64_t, pw, 0, out.At(4)) {
            int64_t wstart = pw * strides.at(2) - padding_before.at(2);
            int64_t wend = std::min(wstart + pool_size.at(2), in.At(4));
            wstart = std::max(wstart, static_cast<int64_t>(0));
            const int64_t pool_index = ((n * out.At(2) + pd) * out.At(3) + ph) * out.At(4) + pw;
            const int64_t size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            FOR_RANGE(int64_t, d, dstart, dend) {
              FOR_RANGE(int64_t, h, hstart, hend) {
                FOR_RANGE(int64_t, w, wstart, wend) {
                  const int64_t input_index = ((n * in.At(2) + d) * in.At(3) + h) * in.At(4) + w;
                  process(pool_index, input_index, size, out_mat, in_mat, out_diff_mat,
                          in_diff_mat);
                }
              }
            }
          }
        }
      }
    }
  }

  static void AvgFWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto* pool_state = dynamic_cast<PoolOpKernelState*>(state);
    CHECK(pool_state != nullptr);
    pool_state->Update(x->shape());
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_first") {
      CFirstForward(pool_state->GetParams3D(), x, y, GetZeroVal<T>,
                    [](const T& lhs, T& rhs) { rhs += lhs; },
                    [](const int64_t size, T& out) { out /= size; });
    } else if (data_format == "channels_last") {
      CLastForward(pool_state->GetParams3D(), x, y, GetZeroVal<T>,
                   [](const int64_t in_col, const int64_t out_col, ConstEigenMatrixMap<T>& in_mat,
                      EigenMatrixMap<T>& out_mat) { out_mat.col(out_col) += in_mat.col(in_col); },
                   [](const int64_t size, const int64_t col, EigenMatrixMap<T>& out_mat) {
                     out_mat.col(col) /= size;
                   });
    } else {
      UNIMPLEMENTED();
    }
  }

  static void AvgBWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto* pool_state = dynamic_cast<PoolOpKernelState*>(state);
    CHECK(pool_state != nullptr);
    pool_state->Update(x->shape());
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_first") {
      CFirstBackward(pool_state->GetParams3D(), dy, y, x, dx,
                     [](const T& in, const T& out, const T& out_diff, const int64_t size,
                        T& in_diff) { in_diff += (out_diff / static_cast<T>(size)); });
    } else if (data_format == "channels_last") {
      CLastBackward(pool_state->GetParams3D(), dy, y, x, dx,
                    [](const int64_t out_col, const int64_t in_col, const int64_t size,
                       ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
                       ConstEigenArrayMap<T>& out_diff_arr, EigenArrayMap<T>& in_diff_arr) {
                      in_diff_arr.col(in_col) += out_diff_arr.col(out_col) / static_cast<T>(size);
                    });
    } else {
      UNIMPLEMENTED();
    }
  }

  static void MaxFWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto* pool_state = dynamic_cast<PoolOpKernelState*>(state);
    CHECK(pool_state != nullptr);
    pool_state->Update(x->shape());
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_first") {
      CFirstForward(pool_state->GetParams3D(), x, y, GetMinVal<T>,
                    [](const T& lhs, T& rhs) {
                      if (lhs > rhs) { rhs = lhs; }
                    },
                    [](const int64_t size, T& out) {});
    } else if (data_format == "channels_last") {
      CLastForward(pool_state->GetParams3D(), x, y, GetMinVal<T>,
                   [](const int64_t in_col, const int64_t out_col, ConstEigenMatrixMap<T>& in_mat,
                      EigenMatrixMap<T>& out_mat) {
                     out_mat.col(out_col) = out_mat.col(out_col).cwiseMax(in_mat.col(in_col));
                   },
                   [](const int64_t size, const int64_t col, EigenMatrixMap<T>& out_mat) {});
    } else {
      UNIMPLEMENTED();
    }
  }

  static void MaxBWCompute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto* pool_state = dynamic_cast<PoolOpKernelState*>(state);
    CHECK(pool_state != nullptr);
    pool_state->Update(x->shape());
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_first") {
      CFirstBackward(
          pool_state->GetParams3D(), dy, y, x, dx,
          [](const T& in, const T& out, const T& out_diff, const int64_t size, T& in_diff) {
            if (in == out) { in_diff += out_diff; }
          });
    } else if (data_format == "channels_last") {
      CLastBackward(
          pool_state->GetParams3D(), dy, y, x, dx,
          [](const int64_t out_col, const int64_t in_col, const int64_t size,
             ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
             ConstEigenArrayMap<T>& out_diff_arr, EigenArrayMap<T>& in_diff_arr) {
            in_diff_arr.col(in_col) +=
                out_diff_arr.col(out_col)
                * (in_arr.col(in_col).cwiseEqual(out_arr.col(out_col)).template cast<T>());
          });
    } else {
      UNIMPLEMENTED();
    }
  }
};

std::shared_ptr<user_op::OpKernelState> DoCreateOpKernelState(user_op::KernelInitContext* ctx,
                                                              const int32_t& dim) {
  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  const std::string& padding = ctx->Attr<std::string>("padding");
  const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
  const auto& padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
  const std::vector<int32_t>& pool_size = ctx->Attr<std::vector<int32_t>>("pool_size");
  const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
  const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
  bool is_dynamic = ctx->TensorDesc4ArgNameAndIndex("x", 0)->is_dynamic();
  Params3D params_3d = Params3D(dim, x_shape, data_format, padding, padding_before, padding_after,
                                pool_size, strides, ceil_mode);
  std::shared_ptr<PoolOpKernelState> state(new PoolOpKernelState(params_3d, is_dynamic));
  return std::move(state);
}

}  // namespace

template<typename T>
class AvgPool1DCpuKernel final : public user_op::OpKernel {
 public:
  AvgPool1DCpuKernel() = default;
  ~AvgPool1DCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx, 1);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolCpuKernelUtil<T>::AvgFWCompute(ctx, state);
  };
};

template<typename T>
class AvgPool1DGradCpuKernel final : public user_op::OpKernel {
 public:
  AvgPool1DGradCpuKernel() = default;
  ~AvgPool1DGradCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx, 1);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolCpuKernelUtil<T>::AvgBWCompute(ctx, state);
  };
};

template<typename T>
class AvgPool2DCpuKernel final : public user_op::OpKernel {
 public:
  AvgPool2DCpuKernel() = default;
  ~AvgPool2DCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx, 2);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolCpuKernelUtil<T>::AvgFWCompute(ctx, state);
  };
};

template<typename T>
class AvgPool2DGradCpuKernel final : public user_op::OpKernel {
 public:
  AvgPool2DGradCpuKernel() = default;
  ~AvgPool2DGradCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx, 2);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolCpuKernelUtil<T>::AvgBWCompute(ctx, state);
  };
};

template<typename T>
class AvgPool3DCpuKernel final : public user_op::OpKernel {
 public:
  AvgPool3DCpuKernel() = default;
  ~AvgPool3DCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx, 3);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolCpuKernelUtil<T>::AvgFWCompute(ctx, state);
  };
};

template<typename T>
class AvgPool3DGradCpuKernel final : public user_op::OpKernel {
 public:
  AvgPool3DGradCpuKernel() = default;
  ~AvgPool3DGradCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx, 3);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolCpuKernelUtil<T>::AvgBWCompute(ctx, state);
  };
};

template<typename T>
class MaxPool1DCpuKernel final : public user_op::OpKernel {
 public:
  MaxPool1DCpuKernel() = default;
  ~MaxPool1DCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx, 1);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolCpuKernelUtil<T>::MaxFWCompute(ctx, state);
  };
};

template<typename T>
class MaxPool1DGradCpuKernel final : public user_op::OpKernel {
 public:
  MaxPool1DGradCpuKernel() = default;
  ~MaxPool1DGradCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx, 1);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolCpuKernelUtil<T>::MaxBWCompute(ctx, state);
  };
};

template<typename T>
class MaxPool2DCpuKernel final : public user_op::OpKernel {
 public:
  MaxPool2DCpuKernel() = default;
  ~MaxPool2DCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx, 2);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolCpuKernelUtil<T>::MaxFWCompute(ctx, state);
  };
};

template<typename T>
class MaxPool2DGradCpuKernel final : public user_op::OpKernel {
 public:
  MaxPool2DGradCpuKernel() = default;
  ~MaxPool2DGradCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx, 2);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolCpuKernelUtil<T>::MaxBWCompute(ctx, state);
  };
};

template<typename T>
class MaxPool3DCpuKernel final : public user_op::OpKernel {
 public:
  MaxPool3DCpuKernel() = default;
  ~MaxPool3DCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx, 3);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolCpuKernelUtil<T>::MaxFWCompute(ctx, state);
  };
};

template<typename T>
class MaxPool3DGradCpuKernel final : public user_op::OpKernel {
 public:
  MaxPool3DGradCpuKernel() = default;
  ~MaxPool3DGradCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateOpKernelState(ctx, 3);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolCpuKernelUtil<T>::MaxBWCompute(ctx, state);
  };
};

#define REGISTER_POOL_CPU_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("avg_pool_1d")                                                  \
      .SetCreateFn<AvgPool1DCpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avg_pool_1d_grad")                                             \
      .SetCreateFn<AvgPool1DGradCpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avg_pool_2d")                                                  \
      .SetCreateFn<AvgPool2DCpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avg_pool_2d_grad")                                             \
      .SetCreateFn<AvgPool2DGradCpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avg_pool_3d")                                                  \
      .SetCreateFn<AvgPool3DCpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avg_pool_3d_grad")                                             \
      .SetCreateFn<AvgPool3DGradCpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_pool_1d")                                                  \
      .SetCreateFn<MaxPool1DCpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_pool_1d_grad")                                             \
      .SetCreateFn<MaxPool1DGradCpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_pool_2d")                                                  \
      .SetCreateFn<MaxPool2DCpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_pool_2d_grad")                                             \
      .SetCreateFn<MaxPool2DGradCpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_pool_3d")                                                  \
      .SetCreateFn<MaxPool3DCpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_pool_3d_grad")                                             \
      .SetCreateFn<MaxPool3DGradCpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_POOL_CPU_KERNEL(float)
REGISTER_POOL_CPU_KERNEL(double)

}  // namespace oneflow
