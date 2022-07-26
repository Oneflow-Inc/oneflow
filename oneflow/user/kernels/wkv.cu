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
#include <stdio.h>
#include <assert.h>
#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/user/kernels/upsample_kernel.h"

namespace oneflow {

namespace {

#define MIN_VALUE (-1e38)
#define Tmax (1024)

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
  F w = _w[_c];
  const F* __restrict__ const k = _k + _offset;
  const F* __restrict__ const v = _v + _offset;
  F* __restrict__ const y = _y + _offset;

  F p = 0, q = 0, o = MIN_VALUE;
  // p and q are running sums divided by exp(o) (to avoid overflows)
  for (int i = 0; i < T; i++) {
    const int ii = i * C;

    F no = max(o, u + k[ii]);
    F A = exp(o - no);
    F B = exp(u + k[ii] - no);
    y[ii] = (A * p + B * v[ii]) / (A * q + B);

    no = max(w + o, k[ii]);
    A = exp(w + o - no);
    B = exp(k[ii] - no);
    p = A * p + B * v[ii];
    q = A * q + B;
    o = no;
  }
}

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
  F w = _w[_c];
  const F* __restrict__ const k = _k + _offset;
  const F* __restrict__ const v = _v + _offset;
  const F* __restrict__ const gy = _gy + _offset;

  F* __restrict__ const gk = _gk + _offset;
  F* __restrict__ const gv = _gv + _offset;

  F y[Tmax], z[Tmax], zexp[Tmax];

  F gw = 0, gu = 0;
  F p = 0, q = 0;
  F dpdw = 0, dqdw = 0;
  F o = MIN_VALUE;
  for (int i = 0; i < T; i++) {
    const int ii = i * C;
    F no = max(o, k[ii] + u);
    F A = exp(o - no);
    F B = exp(k[ii] + u - no);

    F num = A * p + B * v[ii];
    F iden = 1 / (A * q + B);

    y[i] = num * iden;
    z[i] = iden;
    zexp[i] = k[ii] + u - no;

    gw += gy[ii] * (dpdw - dqdw * y[i]) * iden * A;
    gu += gy[ii] * (v[ii] - y[i]) * B * iden;

    no = max(w + o, k[ii]);
    A = exp(w + o - no);
    B = exp(k[ii] - no);
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
    F A = gy[ii] * z[i] * exp(zexp[i]);
    F B = exp(k[ii] + o);
    gk[ii] = A * (v[ii] - y[i]) + B * (gp * v[ii] + gq);
    gv[ii] = A + B * gp;

    F no = max(w + o, zexp[i] - k[ii] - u);
    A = exp(w + o - no);
    B = gy[ii] * z[i] * exp(zexp[i] - k[ii] - u - no);
    gp = A * gp + B;
    gq = A * gq - B * y[i];
    o = no;
  }

  // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass, even
  // though it's not in the forward pass
  const int _offsetBC = _b * C + _c;
  _gw[_offsetBC] += gw * _w[_c];
  _gu[_offsetBC] += gu;
}

template<typename F>
void cuda_forward(const int64_t B, const int64_t T, const int64_t C, const F* w, const F* u,
                  const F* k, const F* v, F* y) {
  dim3 threadsPerBlock(min((int)C, 1024));
  assert(B * C % threadsPerBlock.x == 0);
  dim3 numBlocks(B * C / threadsPerBlock.x);
  kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y);
}

template<typename F>
void cuda_backward(const int64_t B, const int64_t T, const int64_t C, const F* w, const F* u,
                   const F* k, const F* v, const F* gy, F* gw, F* gu, F* gk, F* gv) {
  dim3 threadsPerBlock(min((int)C, 1024));
  assert(B * C % threadsPerBlock.x == 0);
  dim3 numBlocks(B * C / threadsPerBlock.x);
  kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, gy, gw, gu, gk, gv);
}

}  // namespace

template<typename F>
class WkvForwordGPUKernel final : public user_op::OpKernel {
 public:
  WkvForwordGPUKernel() = default;
  ~WkvForwordGPUKernel() = default;

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

    cuda_forward(B, T, C, w->dptr<F>(), u->dptr<F>(), k->dptr<F>(), v->dptr<F>(), y->dptr<F>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_WKVFORWORD_CUDA_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("wkv_forward")                                  \
      .SetCreateFn<WkvForwordGPUKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("w", 0) == GetDataType<dtype>::value));

REGISTER_WKVFORWORD_CUDA_KERNEL(float)
REGISTER_WKVFORWORD_CUDA_KERNEL(double)

template<typename F>
class WkvBackwardGPUKernel final : public user_op::OpKernel {
 public:
  WkvBackwardGPUKernel() = default;
  ~WkvBackwardGPUKernel() = default;

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
    const int64_t B = ctx->Attr<int64_t>("B");
    const int64_t T = ctx->Attr<int64_t>("T");
    const int64_t C = ctx->Attr<int64_t>("C");

    cuda_backward(B, T, C, w->dptr<F>(), u->dptr<F>(), k->dptr<F>(), v->dptr<F>(), gy->dptr<F>(),
                  gw->dptr<F>(), gu->dptr<F>(), gk->dptr<F>(), gv->dptr<F>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_WKVBACKWARD_CUDA_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("wkv_backward")                                 \
      .SetCreateFn<WkvBackwardGPUKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("w", 0) == GetDataType<dtype>::value));

REGISTER_WKVBACKWARD_CUDA_KERNEL(float)
REGISTER_WKVBACKWARD_CUDA_KERNEL(double)

}  // namespace oneflow
