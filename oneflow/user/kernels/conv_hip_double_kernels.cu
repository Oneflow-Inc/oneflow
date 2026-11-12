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
#if defined(WITH_CUDA) && defined(USE_HIP)

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/kernel/new_kernel_util.h"

// MIOpen has no double precision convolution, so on ROCm the double conv, conv data grad, conv
// filter grad and deconv kernels run here instead: im2col (or a gathering col2im) per sample and
// one double GEMM per sample, batched over the groups. They take priority over the cuDNN-based
// kernels for double only.

namespace oneflow {

namespace {

// A convolution from x (n, c, d, h, w) to y (n, f, od, oh, ow), or the channels_last layouts,
// with the spatial axes padded to three dimensions in front.
struct ConvGeometry {
  bool channels_last;
  int64_t n;
  int64_t c;  // x channels
  int64_t f;  // y channels
  int64_t groups;
  int64_t in[3];
  int64_t out[3];
  int64_t kernel[3];
  int64_t stride[3];
  int64_t dilation[3];
  int64_t padding[3];

  OF_DEVICE_FUNC int64_t InSize() const { return in[0] * in[1] * in[2]; }
  OF_DEVICE_FUNC int64_t OutSize() const { return out[0] * out[1] * out[2]; }
  OF_DEVICE_FUNC int64_t KernelSize() const { return kernel[0] * kernel[1] * kernel[2]; }
  OF_DEVICE_FUNC int64_t ChannelsPerGroup() const { return c / groups; }
  OF_DEVICE_FUNC int64_t FiltersPerGroup() const { return f / groups; }
  // Elements of the column buffer of one sample: (c * kernel) x out.
  OF_DEVICE_FUNC int64_t ColSize() const { return c * KernelSize() * OutSize(); }
};

std::vector<int32_t> Pad3(const std::vector<int32_t>& v, int32_t value) {
  std::vector<int32_t> ret(3 - v.size(), value);
  ret.insert(ret.end(), v.begin(), v.end());
  return ret;
}

// `x` and `y` are the shapes of the convolution's input and output, `weight` is (f, c / groups,
// kernel...) or (f, kernel..., c / groups). The padding attribute may list fewer axes than the
// convolution has; the missing leading ones are zero.
template<typename Context>
ConvGeometry MakeConvGeometry(Context* ctx, const ShapeView& x, const ShapeView& y,
                              const ShapeView& weight) {
  ConvGeometry g{};
  g.channels_last = ctx->template Attr<std::string>("data_format") == "channels_last";
  const int ndims = x.NumAxes() - 2;
  CHECK_GE(ndims, 1);
  CHECK_LE(ndims, 3);
  const int spatial = g.channels_last ? 1 : 2;
  g.n = x.At(0);
  g.c = g.channels_last ? x.At(ndims + 1) : x.At(1);
  g.f = g.channels_last ? y.At(ndims + 1) : y.At(1);
  g.groups = ctx->template Attr<int32_t>("groups");
  CHECK_EQ(g.c % g.groups, 0);
  CHECK_EQ(g.f % g.groups, 0);
  const auto strides = Pad3(ctx->template Attr<std::vector<int32_t>>("strides"), 1);
  const auto dilation = Pad3(ctx->template Attr<std::vector<int32_t>>("dilation_rate"), 1);
  const auto padding = Pad3(ctx->template Attr<std::vector<int32_t>>("padding_before"), 0);
  for (int i = 0; i < 3; ++i) {
    const int axis = i - (3 - ndims);
    g.in[i] = axis < 0 ? 1 : x.At(spatial + axis);
    g.out[i] = axis < 0 ? 1 : y.At(spatial + axis);
    g.kernel[i] = axis < 0 ? 1 : weight.At(spatial + axis);
    g.stride[i] = strides[i];
    g.dilation[i] = dilation[i];
    g.padding[i] = padding[i];
  }
  return g;
}

// Column buffer layouts, for one sample:
//   channels_first: col[(c * kernel + k) * out + o]
//   channels_last:  col[o * (kernel * c) + group * (kernel * cg) + k * cg + cg_index]
// so that each group's block is a strided matrix in either layout.
template<typename T>
__global__ void Im2Col(ConvGeometry g, const T* x, T* col) {
  const int64_t kernel = g.KernelSize();
  const int64_t out = g.OutSize();
  const int64_t cg = g.ChannelsPerGroup();
  const int64_t col_size = g.ColSize();
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, col_size) {
    int64_t c, k, o;
    if (g.channels_last) {
      o = i / (kernel * g.c);
      const int64_t r = i % (kernel * g.c);
      const int64_t group = r / (kernel * cg);
      k = (r % (kernel * cg)) / cg;
      c = group * cg + r % cg;
    } else {
      c = i / (kernel * out);
      k = (i / out) % kernel;
      o = i % out;
    }
    const int64_t kd = k / (g.kernel[1] * g.kernel[2]);
    const int64_t kh = (k / g.kernel[2]) % g.kernel[1];
    const int64_t kw = k % g.kernel[2];
    const int64_t od = o / (g.out[1] * g.out[2]);
    const int64_t oh = (o / g.out[2]) % g.out[1];
    const int64_t ow = o % g.out[2];
    const int64_t id = od * g.stride[0] - g.padding[0] + kd * g.dilation[0];
    const int64_t ih = oh * g.stride[1] - g.padding[1] + kh * g.dilation[1];
    const int64_t iw = ow * g.stride[2] - g.padding[2] + kw * g.dilation[2];
    T v = 0;
    if (id >= 0 && id < g.in[0] && ih >= 0 && ih < g.in[1] && iw >= 0 && iw < g.in[2]) {
      const int64_t pos = (id * g.in[1] + ih) * g.in[2] + iw;
      v = g.channels_last ? x[pos * g.c + c] : x[c * g.InSize() + pos];
    }
    col[i] = v;
  }
}

// dx = (accumulate ? dx : 0) + col2im(col). Each dx element gathers the column entries that read
// it, so no atomics are needed and the result is deterministic.
template<typename T>
__global__ void Col2Im(ConvGeometry g, const T* col, bool accumulate, T* dx) {
  const int64_t kernel = g.KernelSize();
  const int64_t out = g.OutSize();
  const int64_t cg = g.ChannelsPerGroup();
  const int64_t in_elems = g.c * g.InSize();
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, in_elems) {
    int64_t c, pos;
    if (g.channels_last) {
      pos = i / g.c;
      c = i % g.c;
    } else {
      c = i / g.InSize();
      pos = i % g.InSize();
    }
    const int64_t id = pos / (g.in[1] * g.in[2]);
    const int64_t ih = (pos / g.in[2]) % g.in[1];
    const int64_t iw = pos % g.in[2];
    T sum = 0;
    for (int64_t kd = 0; kd < g.kernel[0]; ++kd) {
      const int64_t sd = id + g.padding[0] - kd * g.dilation[0];
      if (sd < 0 || sd % g.stride[0] != 0 || sd / g.stride[0] >= g.out[0]) { continue; }
      for (int64_t kh = 0; kh < g.kernel[1]; ++kh) {
        const int64_t sh = ih + g.padding[1] - kh * g.dilation[1];
        if (sh < 0 || sh % g.stride[1] != 0 || sh / g.stride[1] >= g.out[1]) { continue; }
        for (int64_t kw = 0; kw < g.kernel[2]; ++kw) {
          const int64_t sw = iw + g.padding[2] - kw * g.dilation[2];
          if (sw < 0 || sw % g.stride[2] != 0 || sw / g.stride[2] >= g.out[2]) { continue; }
          const int64_t o =
              ((sd / g.stride[0]) * g.out[1] + sh / g.stride[1]) * g.out[2] + sw / g.stride[2];
          const int64_t k = (kd * g.kernel[1] + kh) * g.kernel[2] + kw;
          if (g.channels_last) {
            sum += col[o * (kernel * g.c) + (c / cg) * (kernel * cg) + k * cg + c % cg];
          } else {
            sum += col[(c * kernel + k) * out + o];
          }
        }
      }
    }
    dx[i] = accumulate ? dx[i] + sum : sum;
  }
}

template<typename T>
__global__ void AddBias(int64_t elems, int64_t f, int64_t out, bool channels_last, const T* bias,
                        T* y) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, elems) { y[i] += bias[channels_last ? i % f : (i / out) % f]; }
}

template<typename T>
void Launch(ep::Stream* stream, void (*kernel)(ConvGeometry, const T*, T*), int64_t elems,
            const ConvGeometry& g, const T* in, T* out) {
  if (elems == 0) { return; }
  kernel<<<BlocksNum4ThreadsNum(elems), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(g, in, out);
}

int AsInt(int64_t v) {
  CHECK_LE(v, std::numeric_limits<int>::max());
  return static_cast<int>(v);
}

// Row-major C = op(A) * op(B) for each of `batch` matrices, through the column-major BLAS.
void RowMajorGemm(ep::Stream* stream, bool trans_a, bool trans_b, int64_t m, int64_t n, int64_t k,
                  const double* a, int64_t lda, int64_t stride_a, const double* b, int64_t ldb,
                  int64_t stride_b, double beta, double* c, int64_t ldc, int64_t stride_c,
                  int64_t batch) {
  const double alpha = 1;
  OF_CUBLAS_CHECK(hipblasDgemmStridedBatched(
      stream->As<ep::CudaStream>()->cublas_handle(), trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N,
      trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N, AsInt(n), AsInt(m), AsInt(k), &alpha, b, AsInt(ldb),
      stride_b, a, AsInt(lda), stride_a, &beta, c, AsInt(ldc), stride_c, AsInt(batch)));
}

// y[n] = conv(x[n], w), beta 0 or 1.
void ConvForward(ep::Stream* stream, const ConvGeometry& g, const double* x, const double* w,
                 double beta, double* col, double* y) {
  const int64_t ks = g.KernelSize();
  const int64_t out = g.OutSize();
  const int64_t cg = g.ChannelsPerGroup();
  const int64_t fg = g.FiltersPerGroup();
  for (int64_t n = 0; n < g.n; ++n) {
    Launch<double>(stream, Im2Col<double>, g.ColSize(), g, x + n * g.c * g.InSize(), col);
    double* yn = y + n * g.f * out;
    if (g.channels_last) {
      // y_g (out x fg) = col_g (out x ks * cg) * w_g^T
      RowMajorGemm(stream, false, true, out, fg, ks * cg, col, ks * g.c, ks * cg, w, ks * cg,
                   fg * ks * cg, beta, yn, g.f, fg, g.groups);
    } else {
      // y_g (fg x out) = w_g (fg x cg * ks) * col_g (cg * ks x out)
      RowMajorGemm(stream, false, false, fg, out, cg * ks, w, cg * ks, fg * cg * ks, col, out,
                   cg * ks * out, beta, yn, out, fg * out, g.groups);
    }
  }
}

// dx[n] = (accumulate ? dx[n] : 0) + conv_data_grad(dy[n], w).
void ConvDataGrad(ep::Stream* stream, const ConvGeometry& g, const double* dy, const double* w,
                  bool accumulate, double* col, double* dx) {
  const int64_t ks = g.KernelSize();
  const int64_t out = g.OutSize();
  const int64_t cg = g.ChannelsPerGroup();
  const int64_t fg = g.FiltersPerGroup();
  const int64_t in_elems = g.c * g.InSize();
  for (int64_t n = 0; n < g.n; ++n) {
    const double* dyn = dy + n * g.f * out;
    if (g.channels_last) {
      // col_g (out x ks * cg) = dy_g (out x fg) * w_g (fg x ks * cg)
      RowMajorGemm(stream, false, false, out, ks * cg, fg, dyn, g.f, fg, w, ks * cg, fg * ks * cg,
                   0, col, ks * g.c, ks * cg, g.groups);
    } else {
      // col_g (cg * ks x out) = w_g^T (cg * ks x fg) * dy_g (fg x out)
      RowMajorGemm(stream, true, false, cg * ks, out, fg, w, cg * ks, fg * cg * ks, dyn, out,
                   fg * out, 0, col, out, cg * ks * out, g.groups);
    }
    if (in_elems == 0) { continue; }
    Col2Im<double>
        <<<BlocksNum4ThreadsNum(in_elems), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(g, col, accumulate, dx + n * in_elems);
  }
}

// dw = sum over n of conv_filter_grad(x[n], dy[n]).
void ConvFilterGrad(ep::Stream* stream, const ConvGeometry& g, const double* x, const double* dy,
                    double* col, double* dw) {
  const int64_t ks = g.KernelSize();
  const int64_t out = g.OutSize();
  const int64_t cg = g.ChannelsPerGroup();
  const int64_t fg = g.FiltersPerGroup();
  for (int64_t n = 0; n < g.n; ++n) {
    Launch<double>(stream, Im2Col<double>, g.ColSize(), g, x + n * g.c * g.InSize(), col);
    const double* dyn = dy + n * g.f * out;
    const double beta = n == 0 ? 0 : 1;
    if (g.channels_last) {
      // dw_g (fg x ks * cg) += dy_g^T (fg x out) * col_g (out x ks * cg)
      RowMajorGemm(stream, true, false, fg, ks * cg, out, dyn, g.f, fg, col, ks * g.c, ks * cg,
                   beta, dw, ks * cg, fg * ks * cg, g.groups);
    } else {
      // dw_g (fg x cg * ks) += dy_g (fg x out) * col_g^T (out x cg * ks)
      RowMajorGemm(stream, false, true, fg, cg * ks, out, dyn, out, fg * out, col, out,
                   cg * ks * out, beta, dw, cg * ks, fg * cg * ks, g.groups);
    }
  }
}

size_t ColBufferBytes(const ConvGeometry& g) {
  return std::max<size_t>(g.ColSize() * sizeof(double), 1);
}

class ConvHipDoubleKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    if (in->shape_view().elem_cnt() == 0) { return; }
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ConvGeometry g =
        MakeConvGeometry(ctx, in->shape_view(), out->shape_view(), weight->shape_view());
    double beta = 0;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->shape_view(), out->shape_view());
      Memcpy<DeviceType::kCUDA>(ctx->stream(), out->mut_dptr(), add_to_output->dptr(),
                                out->shape_view().elem_cnt() * sizeof(double));
      beta = 1;
    }
    ConvForward(ctx->stream(), g, in->dptr<double>(), weight->dptr<double>(), beta,
                tmp->mut_dptr<double>(), out->mut_dptr<double>());
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    if (bias != nullptr) {
      const int64_t elems = out->shape_view().elem_cnt();
      AddBias<double><<<BlocksNum4ThreadsNum(elems), kCudaThreadsNumPerBlock, 0,
                        ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          elems, g.f, g.OutSize(), g.channels_last, bias->dptr<double>(), out->mut_dptr<double>());
    }
  }
};

#define REGISTER_CONV_HIP_DOUBLE_KERNEL(op_name)                                            \
  REGISTER_USER_KERNEL(#op_name)                                                            \
      .SetCreateFn<ConvHipDoubleKernel>()                                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                      \
                       && (user_op::HobDataType("in", 0) == DataType::kDouble))             \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                         \
        const Shape& in = ctx->InputTensorDesc("in", 0).shape();                            \
        const Shape& weight = ctx->InputTensorDesc("weight", 0).shape();                    \
        const Shape& out = ctx->OutputTensorDesc("out", 0).shape();                         \
        return ColBufferBytes(                                                              \
            MakeConvGeometry(ctx, ShapeView(in), ShapeView(out), ShapeView(weight)));       \
      })                                                                                    \
      .SetInplaceProposalFn(                                                                \
          [](const user_op::InferContext& ctx,                                              \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {        \
            if (ctx.has_input("_add_to_output", 0)) {                                       \
              OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "_add_to_output", 0, true)); \
            }                                                                               \
            return Maybe<void>::Ok();                                                       \
          })                                                                                \
      .SetPriority(user_op::kKernelPriorityOptimized);

REGISTER_CONV_HIP_DOUBLE_KERNEL(conv1d);
REGISTER_CONV_HIP_DOUBLE_KERNEL(conv2d);
REGISTER_CONV_HIP_DOUBLE_KERNEL(conv3d);

class ConvDataGradHipDoubleKernel final : public user_op::OpKernel,
                                          public user_op::CudaGraphSupport {
 public:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* filter = ctx->Tensor4ArgNameAndIndex("filter", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx->shape_view().elem_cnt() == 0) { return; }
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ConvGeometry g =
        MakeConvGeometry(ctx, dx->shape_view(), dy->shape_view(), filter->shape_view());
    bool accumulate = false;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->shape_view(), dx->shape_view());
      Memcpy<DeviceType::kCUDA>(ctx->stream(), dx->mut_dptr(), add_to_output->dptr(),
                                dx->shape_view().elem_cnt() * sizeof(double));
      accumulate = true;
    }
    ConvDataGrad(ctx->stream(), g, dy->dptr<double>(), filter->dptr<double>(), accumulate,
                 tmp->mut_dptr<double>(), dx->mut_dptr<double>());
  }
};

REGISTER_USER_KERNEL("conv_data_grad")
    .SetCreateFn<ConvDataGradHipDoubleKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("dy", 0) == DataType::kDouble))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {
      const Shape& dy = ctx->InputTensorDesc("dy", 0).shape();
      const Shape& filter = ctx->InputTensorDesc("filter", 0).shape();
      const Shape& dx = ctx->OutputTensorDesc("dx", 0).shape();
      return ColBufferBytes(MakeConvGeometry(ctx, ShapeView(dx), ShapeView(dy), ShapeView(filter)));
    })
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.has_input("_add_to_output", 0)) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "_add_to_output", 0, true));
      }
      return Maybe<void>::Ok();
    })
    .SetPriority(user_op::kKernelPriorityOptimized);

class ConvFilterGradHipDoubleKernel final : public user_op::OpKernel,
                                            public user_op::CudaGraphSupport {
 public:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* filter_diff = ctx->Tensor4ArgNameAndIndex("filter_diff", 0);
    if (x->shape_view().elem_cnt() == 0) {
      Memset<DeviceType::kCUDA>(ctx->stream(), filter_diff->mut_dptr(), 0,
                                filter_diff->shape_view().elem_cnt() * sizeof(double));
      return;
    }
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ConvGeometry g =
        MakeConvGeometry(ctx, x->shape_view(), dy->shape_view(), filter_diff->shape_view());
    ConvFilterGrad(ctx->stream(), g, x->dptr<double>(), dy->dptr<double>(), tmp->mut_dptr<double>(),
                   filter_diff->mut_dptr<double>());
  }
};

REGISTER_USER_KERNEL("conv_filter_grad")
    .SetCreateFn<ConvFilterGradHipDoubleKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("dy", 0) == DataType::kDouble))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {
      const Shape& dy = ctx->InputTensorDesc("dy", 0).shape();
      const Shape& x = ctx->InputTensorDesc("x", 0).shape();
      const Shape& filter_diff = ctx->OutputTensorDesc("filter_diff", 0).shape();
      return ColBufferBytes(
          MakeConvGeometry(ctx, ShapeView(x), ShapeView(dy), ShapeView(filter_diff)));
    })
    .SetPriority(user_op::kKernelPriorityOptimized);

// A deconvolution of `in` is the data gradient of the convolution from `out` to `in`.
class DeconvHipDoubleKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    if (in->shape_view().elem_cnt() == 0) { return; }
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ConvGeometry g =
        MakeConvGeometry(ctx, out->shape_view(), in->shape_view(), weight->shape_view());
    ConvDataGrad(ctx->stream(), g, in->dptr<double>(), weight->dptr<double>(), false,
                 tmp->mut_dptr<double>(), out->mut_dptr<double>());
  }
};

#define REGISTER_DECONV_HIP_DOUBLE_KERNEL(op_name)                                    \
  REGISTER_USER_KERNEL(#op_name)                                                      \
      .SetCreateFn<DeconvHipDoubleKernel>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                \
                       && (user_op::HobDataType("in", 0) == DataType::kDouble))       \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                   \
        const Shape& in = ctx->InputTensorDesc("in", 0).shape();                      \
        const Shape& weight = ctx->InputTensorDesc("weight", 0).shape();              \
        const Shape& out = ctx->OutputTensorDesc("out", 0).shape();                   \
        return ColBufferBytes(                                                        \
            MakeConvGeometry(ctx, ShapeView(out), ShapeView(in), ShapeView(weight))); \
      })                                                                              \
      .SetPriority(user_op::kKernelPriorityOptimized);

REGISTER_DECONV_HIP_DOUBLE_KERNEL(deconv1d);
REGISTER_DECONV_HIP_DOUBLE_KERNEL(deconv2d);
REGISTER_DECONV_HIP_DOUBLE_KERNEL(deconv3d);

}  // namespace

}  // namespace oneflow

#endif  // defined(WITH_CUDA) && defined(USE_HIP)
