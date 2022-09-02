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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/ep/include/primitive/fill.h"

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000

namespace oneflow {

namespace {

#define MIN_VALUE (-1e38)

template<typename T>
__inline__ __device__ T Exp(T x);

template<>
__inline__ __device__ float Exp<float>(float x) {
#ifdef OF_WKV_USE_FAST_MATH
  return __expf(x);
#else
  return exp(x);
#endif
}

template<>
__inline__ __device__ double Exp<double>(double x) {
  return exp(x);
}

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_WKV_USE_FAST_MATH
  return __fdividef(a, b);
#else
  return a / b;
#endif
}

template<>
__inline__ __device__ double Div<double>(double a, double b) {
  return a / b;
}

template<>
__inline__ __device__ nv_bfloat16 Div<nv_bfloat16>(nv_bfloat16 a, nv_bfloat16 b) {
  return a / b;
}

template<>
__inline__ __device__ half Div<half>(half a, half b) {
  return a / b;
}

template<typename F>
__global__ void kernel_forward(const int64_t B, const int64_t T, const int64_t C,
                               const F* __restrict__ const _w, const F* __restrict__ const _u,
                               const F* __restrict__ const _k, const F* __restrict__ const _v,
                               F* __restrict__ const _y) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int _b = idx / C;
  const int _c = idx % C;
  const int _offset = _b * T * C + _c;

  F u = _u[_c];
  const F w = -1 * Exp(_w[_c]);
  const F* __restrict__ const k = _k + _offset;
  const F* __restrict__ const v = _v + _offset;
  F* __restrict__ const y = _y + _offset;

  F p = 0, q = 0, o = MIN_VALUE;
  // p and q are running sums divided by exp(o) (to avoid overflows)
  for (int i = 0; i < T; i++) {
    const int ii = i * C;

    F no = max(o, u + k[ii]);
    F A = Exp(o - no);
    F B = Exp(u + k[ii] - no);
    y[ii] = Div((A * p + B * v[ii]), (A * q + B));

    no = max(w + o, k[ii]);
    A = Exp(w + o - no);
    B = Exp(k[ii] - no);
    p = A * p + B * v[ii];
    q = A * q + B;
    o = no;
  }
}

template<>
__global__ void kernel_forward(const int64_t B, const int64_t T, const int64_t C,
                               const half* __restrict__ const _w, const half* __restrict__ const _u,
                               const half* __restrict__ const _k, const half* __restrict__ const _v,
                               half* __restrict__ const _y) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int _b = idx / C;
  const int _c = idx % C;
  const int _offset = _b * T * C + _c;

  float u = static_cast<float>(_u[_c]);
  const float w = -1.0 * (Exp(static_cast<float>(_w[_c])));
  const half* __restrict__ const k = _k + _offset;
  const half* __restrict__ const v = _v + _offset;
  half* __restrict__ const y = _y + _offset;

  float p = 0.0, q = 0.0, o = MIN_VALUE;
  float no, AA, BB;
  // p and q are running sums divided by exp(o) (to avoid overflows)
  for (int i = 0; i < T; i++) {
    const int ii = i * C;

    no = max(o, u + static_cast<float>(k[ii]));
    AA = Exp(o - no);
    BB = Exp(u + static_cast<float>(k[ii]) - no);
    y[ii] = static_cast<half>(Div((AA * p + BB * static_cast<float>(v[ii])), (AA * q + BB)));

    no = max(w + o, static_cast<float>(k[ii]));
    AA = Exp(w + o - no);
    BB = Exp(static_cast<float>(k[ii]) - no);
    p = AA * p + BB * static_cast<float>(v[ii]);
    q = AA * q + BB;
    o = no;
  }
}

#if CUDA_VERSION >= 11000
template<>
__global__ void kernel_forward(const int64_t B, const int64_t T, const int64_t C,
                               const nv_bfloat16* __restrict__ const _w,
                               const nv_bfloat16* __restrict__ const _u,
                               const nv_bfloat16* __restrict__ const _k,
                               const nv_bfloat16* __restrict__ const _v,
                               nv_bfloat16* __restrict__ const _y) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int _b = idx / C;
  const int _c = idx % C;
  const int _offset = _b * T * C + _c;

  float u = static_cast<float>(_u[_c]);
  const float w = -1.0 * (Exp(static_cast<float>(_w[_c])));
  const nv_bfloat16* __restrict__ const k = _k + _offset;
  const nv_bfloat16* __restrict__ const v = _v + _offset;
  nv_bfloat16* __restrict__ const y = _y + _offset;

  float p = 0.0, q = 0.0, o = MIN_VALUE;
  float no, AA, BB;
  // p and q are running sums divided by exp(o) (to avoid overflows)
  for (int i = 0; i < T; i++) {
    const int ii = i * C;

    no = max(o, u + static_cast<float>(k[ii]));
    AA = Exp(o - no);
    BB = Exp(u + static_cast<float>(k[ii]) - no);
    y[ii] = static_cast<nv_bfloat16>(Div((AA * p + BB * static_cast<float>(v[ii])), (AA * q + BB)));

    no = max(w + o, static_cast<float>(k[ii]));
    AA = Exp(w + o - no);
    BB = Exp(static_cast<float>(k[ii]) - no);
    p = AA * p + BB * static_cast<float>(v[ii]);
    q = AA * q + BB;
    o = no;
  }
}
#endif

template<typename F>
__global__ void kernel_backward(const int64_t B, const int64_t T, const int64_t C,
                                const F* __restrict__ const _w, const F* __restrict__ const _u,
                                const F* __restrict__ const _k, const F* __restrict__ const _v,
                                const F* __restrict__ const _gy, F* __restrict__ const _gw,
                                F* __restrict__ const _gu, F* __restrict__ const _gk,
                                F* __restrict__ const _gv) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int _b = idx / C;
  const int _c = idx % C;
  const int _offset = _b * T * C + _c;

  F u = _u[_c];
  const F w = -1 * Exp(_w[_c]);
  const F* __restrict__ const k = _k + _offset;
  const F* __restrict__ const v = _v + _offset;
  const F* __restrict__ const gy = _gy + _offset;

  F* __restrict__ const gk = _gk + _offset;
  F* __restrict__ const gv = _gv + _offset;

  F y[1024], z[1024], zexp[1024];

  F gw = 0, gu = 0;
  F p = 0, q = 0;
  F dpdw = 0, dqdw = 0;
  F o = MIN_VALUE;
  for (int i = 0; i < T; i++) {
    const int ii = i * C;
    F no = max(o, k[ii] + u);
    F A = Exp(o - no);
    F B = Exp(k[ii] + u - no);

    F num = A * p + B * v[ii];
    F iden = Div(static_cast<F>(1), (A * q + B));

    y[i] = num * iden;
    z[i] = iden;
    zexp[i] = k[ii] + u - no;

    gw += gy[ii] * (dpdw - dqdw * y[i]) * iden * A;
    gu += gy[ii] * (v[ii] - y[i]) * B * iden;

    no = max(w + o, k[ii]);
    A = Exp(w + o - no);
    B = Exp(k[ii] - no);
    dpdw = A * (p + dpdw);
    dqdw = A * (q + dqdw);
    p = A * p + B * v[ii];
    q = A * q + B;
    o = no;
  }

  F gp = 0, gq = 0;
  o = MIN_VALUE;
  for (int i = T - 1; i >= 0; i--) {
    const int ii = i * C;
    F A = gy[ii] * z[i] * Exp(zexp[i]);
    F B = Exp(k[ii] + o);
    gk[ii] = A * (v[ii] - y[i]) + B * (gp * v[ii] + gq);
    gv[ii] = A + B * gp;

    F no = max(w + o, zexp[i] - k[ii] - u);
    A = Exp(w + o - no);
    B = gy[ii] * z[i] * Exp(zexp[i] - k[ii] - u - no);
    gp = A * gp + B;
    gq = A * gq - B * y[i];
    o = no;
  }

  // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass, even
  // though it's not in the forward pass
  const int _offsetBC = _b * C + _c;
  _gw[_offsetBC] += gw * w;
  _gu[_offsetBC] += gu;
}

template<>
__global__ void kernel_backward(const int64_t B, const int64_t T, const int64_t C,
                                const half* __restrict__ const _w,
                                const half* __restrict__ const _u,
                                const half* __restrict__ const _k,
                                const half* __restrict__ const _v,
                                const half* __restrict__ const _gy, half* __restrict__ const _gw,
                                half* __restrict__ const _gu, half* __restrict__ const _gk,
                                half* __restrict__ const _gv) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int _b = idx / C;
  const int _c = idx % C;
  const int _offset = _b * T * C + _c;

  float u = static_cast<float>(_u[_c]);
  const float w = -1 * Exp(static_cast<float>(_w[_c]));
  const half* __restrict__ const k = _k + _offset;
  const half* __restrict__ const v = _v + _offset;
  const half* __restrict__ const gy = _gy + _offset;

  half* __restrict__ const gk = _gk + _offset;
  half* __restrict__ const gv = _gv + _offset;

  float y[1024], z[1024], zexp[1024];

  float gw = 0.0, gu = 0.0;
  float p = 0.0, q = 0.0;
  float dpdw = 0.0, dqdw = 0.0;
  float o = static_cast<float>(MIN_VALUE);
  for (int i = 0; i < T; i++) {
    const int ii = i * C;
    float no = max(o, static_cast<float>(k[ii]) + u);

    float A = Exp(o - no);
    float B = Exp(static_cast<float>(k[ii]) + u - no);

    float num = A * p + B * static_cast<float>(v[ii]);
    float iden = Div(static_cast<float>(1.0), A * q + B);

    y[i] = num * iden;
    z[i] = iden;
    zexp[i] = static_cast<float>(k[ii]) + u - no;

    gw = gw + static_cast<float>(gy[ii]) * (dpdw - dqdw * y[i]) * iden * A;
    gu = gu + static_cast<float>(gy[ii]) * (static_cast<float>(v[ii]) - y[i]) * B * iden;

    no = max(w + o, static_cast<float>(k[ii]));
    A = Exp(w + o - no);
    B = Exp(static_cast<float>(k[ii]) - no);
    dpdw = A * (p + dpdw);
    dqdw = A * (q + dqdw);
    p = A * p + B * static_cast<float>(v[ii]);
    q = A * q + B;
    o = no;
  }

  float gp = 0.0, gq = 0.0;
  o = static_cast<float>(MIN_VALUE);
  for (int i = T - 1; i >= 0; i--) {
    const int ii = i * C;
    float A = static_cast<float>(gy[ii]) * z[i] * Exp(zexp[i]);
    float B = Exp(static_cast<float>(k[ii]) + o);
    gk[ii] = static_cast<half>(A * (static_cast<float>(v[ii]) - y[i])
                               + B * (gp * static_cast<float>(v[ii]) + gq));
    gv[ii] = static_cast<half>(A + B * gp);

    float no = max(w + o, zexp[i] - static_cast<float>(k[ii]) - u);

    A = Exp(w + o - no);
    B = static_cast<float>(gy[ii]) * z[i] * Exp(zexp[i] - static_cast<float>(k[ii]) - u - no);
    gp = A * gp + B;
    gq = A * gq - B * y[i];
    o = no;
  }

  // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass, even
  // though it's not in the forward pass
  const int _offsetBC = _b * C + _c;
  _gw[_offsetBC] = _gw[_offsetBC] + static_cast<half>(gw * w);
  _gu[_offsetBC] = _gu[_offsetBC] + static_cast<half>(gu);
}

#if CUDA_VERSION >= 11000
template<>
__global__ void kernel_backward(
    const int64_t B, const int64_t T, const int64_t C, const nv_bfloat16* __restrict__ const _w,
    const nv_bfloat16* __restrict__ const _u, const nv_bfloat16* __restrict__ const _k,
    const nv_bfloat16* __restrict__ const _v, const nv_bfloat16* __restrict__ const _gy,
    nv_bfloat16* __restrict__ const _gw, nv_bfloat16* __restrict__ const _gu,
    nv_bfloat16* __restrict__ const _gk, nv_bfloat16* __restrict__ const _gv) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int _b = idx / C;
  const int _c = idx % C;
  const int _offset = _b * T * C + _c;

  float u = static_cast<float>(_u[_c]);
  const float w = -1 * Exp(static_cast<float>(_w[_c]));
  const nv_bfloat16* __restrict__ const k = _k + _offset;
  const nv_bfloat16* __restrict__ const v = _v + _offset;
  const nv_bfloat16* __restrict__ const gy = _gy + _offset;

  nv_bfloat16* __restrict__ const gk = _gk + _offset;
  nv_bfloat16* __restrict__ const gv = _gv + _offset;

  float y[1024], z[1024], zexp[1024];

  float gw = 0.0, gu = 0.0;
  float p = 0.0, q = 0.0;
  float dpdw = 0.0, dqdw = 0.0;
  float o = static_cast<float>(MIN_VALUE);
  for (int i = 0; i < T; i++) {
    const int ii = i * C;
    float no = max(o, static_cast<float>(k[ii]) + u);

    float A = Exp(o - no);
    float B = Exp(static_cast<float>(k[ii]) + u - no);

    float num = A * p + B * static_cast<float>(v[ii]);
    float iden = Div(static_cast<float>(1.0), A * q + B);

    y[i] = num * iden;
    z[i] = iden;
    zexp[i] = static_cast<float>(k[ii]) + u - no;

    gw = gw + static_cast<float>(gy[ii]) * (dpdw - dqdw * y[i]) * iden * A;
    gu = gu + static_cast<float>(gy[ii]) * (static_cast<float>(v[ii]) - y[i]) * B * iden;

    no = max(w + o, static_cast<float>(k[ii]));
    A = Exp(w + o - no);
    B = Exp(static_cast<float>(k[ii]) - no);
    dpdw = A * (p + dpdw);
    dqdw = A * (q + dqdw);
    p = A * p + B * static_cast<float>(v[ii]);
    q = A * q + B;
    o = no;
  }

  float gp = 0.0, gq = 0.0;
  o = static_cast<float>(MIN_VALUE);
  for (int i = T - 1; i >= 0; i--) {
    const int ii = i * C;
    float A = static_cast<float>(gy[ii]) * z[i] * Exp(zexp[i]);
    float B = Exp(static_cast<float>(k[ii]) + o);
    gk[ii] = static_cast<nv_bfloat16>(A * (static_cast<float>(v[ii]) - y[i])
                                      + B * (gp * static_cast<float>(v[ii]) + gq));
    gv[ii] = static_cast<nv_bfloat16>(A + B * gp);

    float no = max(w + o, zexp[i] - static_cast<float>(k[ii]) - u);

    A = Exp(w + o - no);
    B = static_cast<float>(gy[ii]) * z[i] * Exp(zexp[i] - static_cast<float>(k[ii]) - u - no);
    gp = A * gp + B;
    gq = A * gq - B * y[i];
    o = no;
  }

  // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass, even
  // though it's not in the forward pass
  const int _offsetBC = _b * C + _c;
  _gw[_offsetBC] = _gw[_offsetBC] + static_cast<nv_bfloat16>(gw * w);
  _gu[_offsetBC] = _gu[_offsetBC] + static_cast<nv_bfloat16>(gu);
}
#endif

template<typename Context>
std::unique_ptr<ep::primitive::Fill> NewFillPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("gy", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->device_type(), data_type);
}

auto FillPrimitiveExists() {
  return hob::make_custom("FillPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewFillPrimitive(&ctx).operator bool();
  });
}

}  // namespace

template<typename F>
class WkvGPUKernel final : public user_op::OpKernel {
 public:
  WkvGPUKernel() = default;
  ~WkvGPUKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* w = ctx->Tensor4ArgNameAndIndex("w", 0);
    const user_op::Tensor* u = ctx->Tensor4ArgNameAndIndex("u", 0);
    const user_op::Tensor* k = ctx->Tensor4ArgNameAndIndex("k", 0);
    const user_op::Tensor* v = ctx->Tensor4ArgNameAndIndex("v", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int64_t B = ctx->Attr<int64_t>("B");
    const int64_t T = ctx->Attr<int64_t>("T");
    const int64_t C = ctx->Attr<int64_t>("C");

    dim3 threadsPerBlock(min((int)C, 32));
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w->dptr<F>(), u->dptr<F>(),
                                                   k->dptr<F>(), v->dptr<F>(), y->mut_dptr<F>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_WKV_CUDA_KERNEL(dtype)                                           \
  REGISTER_USER_KERNEL("wkv").SetCreateFn<WkvGPUKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                             \
      && (user_op::HobDataType("w", 0) == GetDataType<dtype>::value));

#if CUDA_VERSION >= 11000
REGISTER_WKV_CUDA_KERNEL(nv_bfloat16)
#endif  // CUDA_VERSION >= 11000
REGISTER_WKV_CUDA_KERNEL(half)
REGISTER_WKV_CUDA_KERNEL(float)
REGISTER_WKV_CUDA_KERNEL(double)

template<typename F>
class WkvGradGPUKernel final : public user_op::OpKernel {
 public:
  WkvGradGPUKernel() = default;
  ~WkvGradGPUKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* w = ctx->Tensor4ArgNameAndIndex("w", 0);
    const user_op::Tensor* u = ctx->Tensor4ArgNameAndIndex("u", 0);
    const user_op::Tensor* k = ctx->Tensor4ArgNameAndIndex("k", 0);
    const user_op::Tensor* v = ctx->Tensor4ArgNameAndIndex("v", 0);
    const user_op::Tensor* gy = ctx->Tensor4ArgNameAndIndex("gy", 0);
    user_op::Tensor* gw = ctx->Tensor4ArgNameAndIndex("gw", 0);
    user_op::Tensor* gu = ctx->Tensor4ArgNameAndIndex("gu", 0);
    user_op::Tensor* gk = ctx->Tensor4ArgNameAndIndex("gk", 0);
    user_op::Tensor* gv = ctx->Tensor4ArgNameAndIndex("gv", 0);

    Scalar init_value = Scalar(0.0);
    std::unique_ptr<ep::primitive::Fill> fill = NewFillPrimitive(ctx);
    CHECK(fill);
    fill->Launch(ctx->stream(), gw->mut_dptr<F>(), init_value, gw->shape_view().elem_cnt());
    fill->Launch(ctx->stream(), gu->mut_dptr<F>(), init_value, gu->shape_view().elem_cnt());
    fill->Launch(ctx->stream(), gk->mut_dptr<F>(), init_value, gk->shape_view().elem_cnt());
    fill->Launch(ctx->stream(), gv->mut_dptr<F>(), init_value, gv->shape_view().elem_cnt());
    const int64_t B = ctx->Attr<int64_t>("B");
    const int64_t T = ctx->Attr<int64_t>("T");
    const int64_t C = ctx->Attr<int64_t>("C");

    dim3 threadsPerBlock(min((int)C, 32));
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(
        B, T, C, w->dptr<F>(), u->dptr<F>(), k->dptr<F>(), v->dptr<F>(), gy->dptr<F>(),
        gw->mut_dptr<F>(), gu->mut_dptr<F>(), gk->mut_dptr<F>(), gv->mut_dptr<F>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_WKVGRAD_CUDA_KERNEL(dtype)                                           \
  REGISTER_USER_KERNEL("wkv_grad")                                                    \
      .SetCreateFn<WkvGradGPUKernel<dtype>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                \
                       && (user_op::HobDataType("w", 0) == GetDataType<dtype>::value) \
                       && FillPrimitiveExists());

#if CUDA_VERSION >= 11000
REGISTER_WKVGRAD_CUDA_KERNEL(nv_bfloat16)
#endif  // CUDA_VERSION >= 11000
REGISTER_WKVGRAD_CUDA_KERNEL(half)
REGISTER_WKVGRAD_CUDA_KERNEL(float)
REGISTER_WKVGRAD_CUDA_KERNEL(double)

}  // namespace oneflow
