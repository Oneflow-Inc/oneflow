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

template<typename E, typename F>
__global__ void kernel_forward(const int64_t B, const int64_t T, const int64_t C,
                               const E* __restrict__ const _w, const E* __restrict__ const _u,
                               const E* __restrict__ const _k, const E* __restrict__ const _v,
                               E* __restrict__ const _y) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int _b = idx / C;
  const int _c = idx % C;
  const int _offset = _b * T * C + _c;

  F u = static_cast<F>(_u[_c]);
  const F w = -1.0 * (Exp(static_cast<F>(_w[_c])));
  const E* __restrict__ const k = _k + _offset;
  const E* __restrict__ const v = _v + _offset;
  E* __restrict__ const y = _y + _offset;

  F p = 0.0, q = 0.0, o = MIN_VALUE;
  F no, AA, BB;
  // p and q are running sums divided by exp(o) (to avoid overflows)
  for (int i = 0; i < T; i++) {
    const int ii = i * C;

    no = max(o, u + static_cast<F>(k[ii]));
    AA = Exp(o - no);
    BB = Exp(u + static_cast<F>(k[ii]) - no);
    y[ii] = static_cast<E>(Div((AA * p + BB * static_cast<F>(v[ii])), (AA * q + BB)));

    no = max(w + o, static_cast<F>(k[ii]));
    AA = Exp(w + o - no);
    BB = Exp(static_cast<F>(k[ii]) - no);
    p = AA * p + BB * static_cast<F>(v[ii]);
    q = AA * q + BB;
    o = no;
  }
}

template<typename E, typename F>
__global__ void kernel_backward(const int64_t B, const int64_t T, const int64_t C,
                                const E* __restrict__ const _w, const E* __restrict__ const _u,
                                const E* __restrict__ const _k, const E* __restrict__ const _v,
                                const E* __restrict__ const _gy, E* __restrict__ const _gw,
                                E* __restrict__ const _gu, E* __restrict__ const _gk,
                                E* __restrict__ const _gv) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int _b = idx / C;
  const int _c = idx % C;
  const int _offset = _b * T * C + _c;

  F u = static_cast<F>(_u[_c]);
  const F w = -1.0 * (Exp(static_cast<F>(_w[_c])));
  const E* __restrict__ const k = _k + _offset;
  const E* __restrict__ const v = _v + _offset;
  const E* __restrict__ const gy = _gy + _offset;

  E* __restrict__ const gk = _gk + _offset;
  E* __restrict__ const gv = _gv + _offset;

  F y[1024], z[1024], zexp[1024];

  F gw = 0.0, gu = 0.0;
  F p = 0.0, q = 0.0;
  F dpdw = 0.0, dqdw = 0.0;
  F o = static_cast<F>(MIN_VALUE);
  for (int i = 0; i < T; i++) {
    const int ii = i * C;
    F no = max(o, static_cast<F>(k[ii]) + u);

    F A = Exp(o - no);
    F B = Exp(static_cast<F>(k[ii]) + u - no);

    F num = A * p + B * static_cast<F>(v[ii]);
    F iden = Div(static_cast<F>(1.0), A * q + B);

    y[i] = num * iden;
    z[i] = iden;
    zexp[i] = static_cast<F>(k[ii]) + u - no;

    gw = gw + static_cast<F>(gy[ii]) * (dpdw - dqdw * y[i]) * iden * A;
    gu = gu + static_cast<F>(gy[ii]) * (static_cast<F>(v[ii]) - y[i]) * B * iden;

    no = max(w + o, static_cast<F>(k[ii]));
    A = Exp(w + o - no);
    B = Exp(static_cast<F>(k[ii]) - no);
    dpdw = A * (p + dpdw);
    dqdw = A * (q + dqdw);
    p = A * p + B * static_cast<F>(v[ii]);
    q = A * q + B;
    o = no;
  }

  F gp = 0.0, gq = 0.0;
  o = static_cast<F>(MIN_VALUE);
  for (int i = T - 1; i >= 0; i--) {
    const int ii = i * C;
    F A = static_cast<F>(gy[ii]) * z[i] * Exp(zexp[i]);
    F B = Exp(static_cast<F>(k[ii]) + o);
    gk[ii] =
        static_cast<E>(A * (static_cast<F>(v[ii]) - y[i]) + B * (gp * static_cast<F>(v[ii]) + gq));
    gv[ii] = static_cast<E>(A + B * gp);

    F no = max(w + o, zexp[i] - static_cast<F>(k[ii]) - u);

    A = Exp(w + o - no);
    B = static_cast<F>(gy[ii]) * z[i] * Exp(zexp[i] - static_cast<F>(k[ii]) - u - no);
    gp = A * gp + B;
    gq = A * gq - B * y[i];
    o = no;
  }

  // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass, even
  // though it's not in the forward pass
  const int _offsetBC = _b * C + _c;
  _gw[_offsetBC] = static_cast<E>(static_cast<F>(_gw[_offsetBC]) + gw * w);
  _gu[_offsetBC] = static_cast<E>(static_cast<F>(_gu[_offsetBC]) + gu);
}

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

template<typename E, typename F>
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
    kernel_forward<E, F><<<numBlocks, threadsPerBlock>>>(
        B, T, C, w->dptr<E>(), u->dptr<E>(), k->dptr<E>(), v->dptr<E>(), y->mut_dptr<E>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_WKV_CUDA_KERNEL(e_dtype, f_dtype)                                           \
  REGISTER_USER_KERNEL("wkv").SetCreateFn<WkvGPUKernel<e_dtype, f_dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                        \
      && (user_op::HobDataType("w", 0) == GetDataType<e_dtype>::value));

#if CUDA_VERSION >= 11000
REGISTER_WKV_CUDA_KERNEL(nv_bfloat16, float)
#endif  // CUDA_VERSION >= 11000
REGISTER_WKV_CUDA_KERNEL(half, float)
REGISTER_WKV_CUDA_KERNEL(float, float)
REGISTER_WKV_CUDA_KERNEL(double, double)

template<typename E, typename F>
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
    fill->Launch(ctx->stream(), gw->mut_dptr<E>(), init_value, gw->shape_view().elem_cnt());
    fill->Launch(ctx->stream(), gu->mut_dptr<E>(), init_value, gu->shape_view().elem_cnt());
    fill->Launch(ctx->stream(), gk->mut_dptr<E>(), init_value, gk->shape_view().elem_cnt());
    fill->Launch(ctx->stream(), gv->mut_dptr<E>(), init_value, gv->shape_view().elem_cnt());
    const int64_t B = ctx->Attr<int64_t>("B");
    const int64_t T = ctx->Attr<int64_t>("T");
    const int64_t C = ctx->Attr<int64_t>("C");

    dim3 threadsPerBlock(min((int)C, 32));
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<E, F><<<numBlocks, threadsPerBlock>>>(
        B, T, C, w->dptr<E>(), u->dptr<E>(), k->dptr<E>(), v->dptr<E>(), gy->dptr<E>(),
        gw->mut_dptr<E>(), gu->mut_dptr<E>(), gk->mut_dptr<E>(), gv->mut_dptr<E>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_WKVGRAD_CUDA_KERNEL(e_dtype, f_dtype)                                  \
  REGISTER_USER_KERNEL("wkv_grad")                                                      \
      .SetCreateFn<WkvGradGPUKernel<e_dtype, f_dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("w", 0) == GetDataType<e_dtype>::value) \
                       && FillPrimitiveExists());

#if CUDA_VERSION >= 11000
REGISTER_WKVGRAD_CUDA_KERNEL(nv_bfloat16, float)
#endif  // CUDA_VERSION >= 11000
REGISTER_WKVGRAD_CUDA_KERNEL(half, float)
REGISTER_WKVGRAD_CUDA_KERNEL(float, double)
REGISTER_WKVGRAD_CUDA_KERNEL(double, double)

}  // namespace oneflow
