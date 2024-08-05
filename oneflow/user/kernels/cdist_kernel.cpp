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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/framework/user_op_hob.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

template<typename T>
struct ZeroDist {
  static inline T map(const T& diff, const double& p) { return diff == T(0) ? diff : T(1); }
  static inline T reduce(const T& agg, const double& up) { return agg + up; }
  static inline T finish(const T agg, const double p) { return agg; }
  // backward always return 0
};

template<typename T>
struct OneDist {
  static inline T map(const T& diff, const double& p) { return diff; }
  static inline T reduce(const T& agg, const double& up) { return agg + up; }
  static inline T finish(const T agg, const double p) { return agg; }
  static inline T backward(const T& diff, const T grad, const T dist, const double& p) {
    return grad * (diff > T(0) ? T(1) : T(-1));
  }
};

template<typename T>
struct TwoDist {
  static inline T map(const T& diff, const double& p) { return diff * diff; }
  static inline T reduce(const T& agg, const T& up) { return agg + up; }
  static inline T finish(const T agg, const double p) { return std::sqrt(agg); }
  static inline T backward(const T& diff, const T grad, const T dist, const double& p) {
    return dist == 0.0 ? T(0) : grad * diff / dist;
  }
};

template<typename T>
struct InfiDist {
  static inline T map(const T& diff, const double& p) { return diff; }
  static inline T reduce(const T& agg, const T& up) { return std::max(agg, up); }
  static inline T finish(const T agg, const double p) { return agg; }
  static inline T backward(const T& diff, const T grad, const T dist, const double& p) {
    return (T(1) - std::min(std::ceil(std::abs(std::abs(diff) - dist)), T(1))) * grad
           * (diff > T(0) ? T(1) : T(-1));
  }
};

template<typename T>
struct PDist {
  static inline T map(const T& diff, const double& p) { return std::pow(diff, p); }
  static inline T reduce(const T& agg, const T& up) { return agg + up; }
  static inline T finish(const T agg, const double p) { return std::pow(agg, 1.0 / p); }
  static inline T backward(const T& diff, const T grad, const T dist, const double& p) {
    if (dist == 0.0) {
      return T(0);
    } else {
      return diff * std::pow(std::abs(diff), p - T(2)) * grad / std::pow(dist, p - T(1));
    }
  }
};

template<typename T, typename Dist>
void CpuCDistForward(ep::CpuStream* stream, const T* x1, const T* x2, T* out, int64_t size_out,
                     int64_t r1, int64_t r2, int64_t c, double p) {
  // x1 shape: (d1, d2, ..., dn, r1, c), treated as (d1 * ... * dn, r1 * c)
  // x2 shape: (d1, d2, ..., dn, r2, c), treated as (d1 * ... * dn, r2 * c)
  // out shape: (d1, d2, ..., dn, r1, r2), treated as (d1 * ... * dn, r1 * r2)
  // d = d1 * ... * dn
  stream->ParallelFor(
      0, size_out,
      [x1, x2, out, r1, r2, c, p](int64_t begin, int64_t end) {
        // begin is a multiple of c
        T* out_begin = out + begin;
        const T* out_end = out + end;

        int64_t d = r1 * r2;
        int64_t batch_idx = begin / d;
        int64_t vec_out_idx = begin - d * batch_idx;
        int64_t vec1_idx = (vec_out_idx / r2);
        int64_t vec2_idx = vec_out_idx - vec1_idx * r2;
        int64_t vec1_begin = vec1_idx * c;
        int64_t vec2_begin = vec2_idx * c;
        int64_t size1 = r1 * c;
        int64_t size2 = r2 * c;

        while (out_begin != out_end) {
          T agg = 0;
          const T* x1_begin = x1 + batch_idx * size1 + vec1_begin;
          const T* x2_begin = x2 + batch_idx * size2 + vec2_begin;
          FOR_RANGE(int32_t, idx, 0, c) {
            T a = *(x1_begin + idx);
            T b = *(x2_begin + idx);
            agg = Dist::reduce(agg, Dist::map(std::abs(a - b), p));
          }
          *out_begin = Dist::finish(agg, p);
          out_begin += 1;
          vec2_begin += c;
          if (vec2_begin == r2 * c) {
            vec2_begin = 0;
            vec1_begin += c;
            if (vec1_begin == r1 * c) {
              vec1_begin = 0;
              batch_idx += 1;
            }
          }
        }
      },
      c);
}

template<typename T, typename Dist>
void CpuCDistBackward(ep::CpuStream* stream, const T* x1, const T* x2, const T* dist, const T* grad,
                      T* grad1, T* grad2, int64_t size_out, int64_t r1, int64_t r2, int64_t c,
                      double p) {
  stream->ParallelFor(
      0, size_out,
      [=](int64_t begin, int64_t end) {
        const T* dist_begin = dist + begin;
        const T* dist_end = dist + end;
        const T* dist_grad = grad + begin;

        int64_t d = r1 * r2;
        int64_t batch_idx = begin / d;
        int64_t vec_out_idx = begin - d * batch_idx;
        int64_t vec1_idx = (vec_out_idx / r2);
        int64_t vec2_idx = vec_out_idx - vec1_idx * r2;
        int64_t vec1_begin = vec1_idx * c;
        int64_t vec2_begin = vec2_idx * c;
        int64_t size1 = r1 * c;
        int64_t size2 = r2 * c;

        while (dist_begin != dist_end) {
          const T* x1_begin = x1 + batch_idx * size1 + vec1_begin;
          const T* x2_begin = x2 + batch_idx * size2 + vec2_begin;
          T* x1_grad_begin = grad1 + batch_idx * size1 + vec1_begin;
          T* x2_grad_begin = grad2 + batch_idx * size2 + vec2_begin;
          FOR_RANGE(int32_t, idx, 0, c) {
            T a = *(x1_begin + idx);
            T b = *(x2_begin + idx);
            T diff = a - b;
            *(x1_grad_begin + idx) += Dist::backward(diff, *dist_grad, *dist_begin, p);
            *(x2_grad_begin + idx) += Dist::backward(-diff, *dist_grad, *dist_begin, p);
          }

          dist_begin += 1;
          dist_grad += 1;
          vec2_begin += c;
          if (vec2_begin == r2 * c) {
            vec2_begin = 0;
            vec1_begin += c;
            if (vec1_begin == r1 * c) {
              vec1_begin = 0;
              batch_idx += 1;
            }
          }
        }
      },
      c);
}

template<typename T>
class CpuCDistKernel final : public user_op::OpKernel {
 public:
  CpuCDistKernel() = default;
  ~CpuCDistKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x1 = ctx->Tensor4ArgNameAndIndex("x1", 0);
    const user_op::Tensor* x2 = ctx->Tensor4ArgNameAndIndex("x2", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    double p = ctx->Attr<double>("p");
    int64_t ndim = x1->shape_view().NumAxes();
    int64_t r1 = x1->shape_view().At(ndim - 2);
    int64_t r2 = x2->shape_view().At(ndim - 2);
    int64_t c = x1->shape_view().At(ndim - 1);

    const T* x1_ptr = x1->dptr<T>();
    const T* x2_ptr = x2->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    if (p == 0) {
      CpuCDistForward<T, ZeroDist<T>>(ctx->stream()->As<ep::CpuStream>(), x1_ptr, x2_ptr, out_ptr,
                                      out->shape_view().elem_cnt(), r1, r2, c, p);
    } else if (p == 1) {
      CpuCDistForward<T, OneDist<T>>(ctx->stream()->As<ep::CpuStream>(), x1_ptr, x2_ptr, out_ptr,
                                     out->shape_view().elem_cnt(), r1, r2, c, p);
    } else if (p == 2) {
      CpuCDistForward<T, TwoDist<T>>(ctx->stream()->As<ep::CpuStream>(), x1_ptr, x2_ptr, out_ptr,
                                     out->shape_view().elem_cnt(), r1, r2, c, p);
    } else if (std::isinf(p)) {
      CpuCDistForward<T, InfiDist<T>>(ctx->stream()->As<ep::CpuStream>(), x1_ptr, x2_ptr, out_ptr,
                                      out->shape_view().elem_cnt(), r1, r2, c, p);
    } else {
      CpuCDistForward<T, PDist<T>>(ctx->stream()->As<ep::CpuStream>(), x1_ptr, x2_ptr, out_ptr,
                                   out->shape_view().elem_cnt(), r1, r2, c, p);
    };
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class CpuCDistGradKernel final : public user_op::OpKernel {
 public:
  CpuCDistGradKernel() = default;
  ~CpuCDistGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x1 = ctx->Tensor4ArgNameAndIndex("x1", 0);
    const user_op::Tensor* x2 = ctx->Tensor4ArgNameAndIndex("x2", 0);
    const user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx1 = ctx->Tensor4ArgNameAndIndex("dx1", 0);
    user_op::Tensor* dx2 = ctx->Tensor4ArgNameAndIndex("dx2", 0);
    double p = ctx->Attr<double>("p");
    int64_t ndim = x1->shape_view().NumAxes();
    int64_t r1 = x1->shape_view().At(ndim - 2);
    int64_t r2 = x2->shape_view().At(ndim - 2);
    int64_t c = x1->shape_view().At(ndim - 1);

    const T* x1_ptr = x1->dptr<T>();
    const T* x2_ptr = x2->dptr<T>();
    const T* dist_ptr = out->dptr<T>();
    const T* grad_ptr = dy->dptr<T>();

    T* dx1_ptr = dx1->mut_dptr<T>();
    T* dx2_ptr = dx2->mut_dptr<T>();

    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
    CHECK(memset_primitive);
    memset_primitive->Launch(ctx->stream(), dx1_ptr, 0, dx1->shape_view().elem_cnt() * sizeof(T));
    memset_primitive->Launch(ctx->stream(), dx2_ptr, 0, dx2->shape_view().elem_cnt() * sizeof(T));

    if (p == 0) {
      // grad is always zero
    } else if (p == 1) {
      CpuCDistBackward<T, OneDist<T>>(ctx->stream()->As<ep::CpuStream>(), x1_ptr, x2_ptr, dist_ptr,
                                      grad_ptr, dx1_ptr, dx2_ptr, out->shape_view().elem_cnt(), r1,
                                      r2, c, p);
    } else if (p == 2) {
      CpuCDistBackward<T, TwoDist<T>>(ctx->stream()->As<ep::CpuStream>(), x1_ptr, x2_ptr, dist_ptr,
                                      grad_ptr, dx1_ptr, dx2_ptr, out->shape_view().elem_cnt(), r1,
                                      r2, c, p);
    } else if (std::isinf(p)) {
      CpuCDistBackward<T, InfiDist<T>>(ctx->stream()->As<ep::CpuStream>(), x1_ptr, x2_ptr, dist_ptr,
                                       grad_ptr, dx1_ptr, dx2_ptr, out->shape_view().elem_cnt(), r1,
                                       r2, c, p);
    } else {
      CpuCDistBackward<T, PDist<T>>(ctx->stream()->As<ep::CpuStream>(), x1_ptr, x2_ptr, dist_ptr,
                                    grad_ptr, dx1_ptr, dx2_ptr, out->shape_view().elem_cnt(), r1,
                                    r2, c, p);
    };
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_CDIST_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("cdist").SetCreateFn<CpuCDistKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                  \
      && (user_op::HobDataType("x1", 0) == GetDataType<dtype>::value)                 \
      && (user_op::HobDataType("x2", 0) == GetDataType<dtype>::value)                 \
      && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_CDIST_KERNEL(float)
REGISTER_CPU_CDIST_KERNEL(double)
#undef REGISTER_CPU_CDIST_KERNEL

#define REGISTER_CPU_CDIST_GRAD_KERNEL(dtype)                                          \
  REGISTER_USER_KERNEL("cdist_grad")                                                   \
      .SetCreateFn<CpuCDistGradKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                  \
                       && (user_op::HobDataType("x1", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("x2", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_CDIST_GRAD_KERNEL(float)
REGISTER_CPU_CDIST_GRAD_KERNEL(double)
#undef REGISTER_CPU_CDIST_KERNEL

}  // namespace oneflow
