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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/user/kernels/upsample_kernel.h"

namespace oneflow {

namespace {

template<typename T>
static void UpsampleTrilinear3DForward(const int64_t elem_cnt, const T* in_dptr,
                                       NdIndexOffsetHelper<int64_t, 5> in_helper,
                                       NdIndexOffsetHelper<int64_t, 5> out_helper,
                                       const int64_t in_depth, const int64_t in_height,
                                       const int64_t in_width, const T rdepth, const T rheight,
                                       const T rwidth, const bool align_corners, T* out_dptr) {
  for (int64_t index = 0; index < elem_cnt; ++index) {
    int64_t n, c, d, h, w;
    out_helper.OffsetToNdIndex(index, n, c, d, h, w);

    const T t1r = GetAreaPixel(rdepth, d, align_corners);
    const int64_t t1 = t1r;
    const int64_t t1p = (t1 < in_depth - 1) ? 1 : 0;
    const T t1lambda = t1r - t1;
    const T t0lambda = static_cast<T>(1.) - t1lambda;

    const T h1r = GetAreaPixel(rheight, h, align_corners);
    const int64_t h1 = h1r;
    const int64_t h1p = (h1 < in_height - 1) ? 1 : 0;
    const T h1lambda = h1r - h1;
    const T h0lambda = static_cast<T>(1.) - h1lambda;

    const T w1r = GetAreaPixel(rwidth, w, align_corners);
    const int64_t w1 = w1r;
    const int64_t w1p = (w1 < in_width - 1) ? 1 : 0;
    const T w1lambda = w1r - w1;
    const T w0lambda = static_cast<T>(1.) - w1lambda;

    const T* pos1 = &in_dptr[in_helper.NdIndexToOffset(n, c, t1, h1, w1)];

    out_dptr[index] =
        t0lambda
            * (h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p])
               + h1lambda
                     * (w0lambda * pos1[h1p * in_width] + w1lambda * pos1[h1p * in_width + w1p]))
        + t1lambda
              * (h0lambda
                     * (w0lambda * pos1[t1p * in_height * in_width]
                        + w1lambda * pos1[t1p * in_height * in_width + w1p])
                 + h1lambda
                       * (w0lambda * pos1[t1p * in_height * in_width + h1p * in_width]
                          + w1lambda * pos1[t1p * in_height * in_width + h1p * in_width + w1p]));
  }
}

template<typename T>
static void UpsampleTrilinear3DBackward(const int64_t elem_cnt, const T* dy_dptr,
                                        NdIndexOffsetHelper<int64_t, 5> dy_helper,
                                        NdIndexOffsetHelper<int64_t, 5> dx_helper,
                                        const int64_t in_depth, const int64_t in_height,
                                        const int64_t in_width, const T rdepth, const T rheight,
                                        const T rwidth, const bool align_corners, T* dx_dptr) {
  for (int64_t index = 0; index < elem_cnt; ++index) {
    int64_t n, c, d, h, w;
    dy_helper.OffsetToNdIndex(index, n, c, d, h, w);

    const T t1r = GetAreaPixel(rdepth, d, align_corners);
    const int64_t t1 = t1r;
    const int64_t t1p = (t1 < in_depth - 1) ? 1 : 0;
    const T t1lambda = t1r - t1;
    const T t0lambda = static_cast<T>(1.) - t1lambda;

    const T h1r = GetAreaPixel(rheight, h, align_corners);
    const int64_t h1 = h1r;
    const int64_t h1p = (h1 < in_height - 1) ? 1 : 0;
    const T h1lambda = h1r - h1;
    const T h0lambda = static_cast<T>(1.) - h1lambda;

    const T w1r = GetAreaPixel(rwidth, w, align_corners);
    const int64_t w1 = w1r;
    const int64_t w1p = (w1 < in_width - 1) ? 1 : 0;
    const T w1lambda = w1r - w1;
    const T w0lambda = static_cast<T>(1.) - w1lambda;

    T* pos1 = &dx_dptr[dx_helper.NdIndexToOffset(n, c, t1, h1, w1)];
    const T* pos2 = &dy_dptr[index];

    pos1[0] += t0lambda * h0lambda * w0lambda * pos2[0];
    pos1[w1p] += t0lambda * h0lambda * w1lambda * pos2[0];
    pos1[h1p * in_width] += t0lambda * h1lambda * w0lambda * pos2[0];
    pos1[h1p * in_width + w1p] += t0lambda * h1lambda * w1lambda * pos2[0];
    pos1[t1p * in_height * in_width] += t1lambda * h0lambda * w0lambda * pos2[0];
    pos1[t1p * in_height * in_width + w1p] += t1lambda * h0lambda * w1lambda * pos2[0];
    pos1[t1p * in_height * in_width + h1p * in_width] += t1lambda * h1lambda * w0lambda * pos2[0];
    pos1[t1p * in_height * in_width + h1p * in_width + w1p] +=
        t1lambda * h1lambda * w1lambda * pos2[0];
  }
}

}  // namespace

template<typename T>
class UpsampleTrilinear3DCPUKernel final : public user_op::OpKernel {
 public:
  UpsampleTrilinear3DCPUKernel() = default;
  ~UpsampleTrilinear3DCPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const bool align_corners = ctx->Attr<bool>("align_corners");
    const int64_t elem_cnt = y_tensor->shape_view().elem_cnt();
    NdIndexOffsetHelper<int64_t, 5> in_helper(
        x_tensor->shape_view().At(0), x_tensor->shape_view().At(1), x_tensor->shape_view().At(2),
        x_tensor->shape_view().At(3), x_tensor->shape_view().At(4));
    NdIndexOffsetHelper<int64_t, 5> out_helper(
        y_tensor->shape_view().At(0), y_tensor->shape_view().At(1), y_tensor->shape_view().At(2),
        y_tensor->shape_view().At(3), y_tensor->shape_view().At(4));

    const int64_t in_depth = x_tensor->shape_view().At(2);
    const int64_t in_height = x_tensor->shape_view().At(3);
    const int64_t in_width = x_tensor->shape_view().At(4);

    const int64_t out_depth = y_tensor->shape_view().At(2);
    const int64_t out_height = y_tensor->shape_view().At(3);
    const int64_t out_width = y_tensor->shape_view().At(4);

    const std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    double depth_scale = ctx->Attr<double>("depth_scale");
    double height_scale = ctx->Attr<double>("height_scale");
    double width_scale = ctx->Attr<double>("width_scale");
    if (!output_size.empty()) {
      depth_scale = static_cast<double>(out_depth) / static_cast<double>(in_depth);
      height_scale = static_cast<double>(out_height) / static_cast<double>(in_height);
      width_scale = static_cast<double>(out_width) / static_cast<double>(in_width);
    }

    const T scale_depth = GetAreaPixelScale(in_depth, out_depth, align_corners, depth_scale);
    const T scale_height = GetAreaPixelScale(in_height, out_height, align_corners, height_scale);
    const T scale_width = GetAreaPixelScale(in_width, out_width, align_corners, width_scale);

    UpsampleTrilinear3DForward<T>(elem_cnt, x_tensor->dptr<T>(), in_helper, out_helper,
                                  x_tensor->shape_view().At(2), x_tensor->shape_view().At(3),
                                  x_tensor->shape_view().At(4), scale_depth, scale_height,
                                  scale_width, align_corners, y_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleTrilinearGrad3DCPUKernel final : public user_op::OpKernel {
 public:
  UpsampleTrilinearGrad3DCPUKernel() = default;
  ~UpsampleTrilinearGrad3DCPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);

    Memset<DeviceType::kCPU>(ctx->stream(), dx_tensor->mut_dptr<T>(), 0,
                             dx_tensor->shape_view().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const bool align_corners = ctx->Attr<bool>("align_corners");
    const int64_t elem_cnt = dy_tensor->shape_view().elem_cnt();
    NdIndexOffsetHelper<int64_t, 5> dy_helper(
        dy_tensor->shape_view().At(0), dy_tensor->shape_view().At(1), dy_tensor->shape_view().At(2),
        dy_tensor->shape_view().At(3), dy_tensor->shape_view().At(4));
    NdIndexOffsetHelper<int64_t, 5> dx_helper(
        dx_tensor->shape_view().At(0), dx_tensor->shape_view().At(1), dx_tensor->shape_view().At(2),
        dx_tensor->shape_view().At(3), dx_tensor->shape_view().At(4));

    const int64_t in_depth = dx_tensor->shape_view().At(2);
    const int64_t in_height = dx_tensor->shape_view().At(3);
    const int64_t in_width = dx_tensor->shape_view().At(4);

    const int64_t out_depth = dy_tensor->shape_view().At(2);
    const int64_t out_height = dy_tensor->shape_view().At(3);
    const int64_t out_width = dy_tensor->shape_view().At(4);

    const std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    double depth_scale = ctx->Attr<double>("depth_scale");
    double height_scale = ctx->Attr<double>("height_scale");
    double width_scale = ctx->Attr<double>("width_scale");
    if (!output_size.empty()) {
      depth_scale = static_cast<double>(out_depth) / static_cast<double>(in_depth);
      height_scale = static_cast<double>(out_height) / static_cast<double>(in_height);
      width_scale = static_cast<double>(out_width) / static_cast<double>(in_width);
    }

    const T scale_depth = GetAreaPixelScale(in_depth, out_depth, align_corners, depth_scale);
    const T scale_height = GetAreaPixelScale(in_height, out_height, align_corners, height_scale);
    const T scale_width = GetAreaPixelScale(in_width, out_width, align_corners, width_scale);

    UpsampleTrilinear3DBackward<T>(elem_cnt, dy_tensor->dptr<T>(), dy_helper, dx_helper,
                                   dx_tensor->shape_view().At(2), dx_tensor->shape_view().At(3),
                                   dx_tensor->shape_view().At(4), scale_depth, scale_height,
                                   scale_width, align_corners, dx_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPTRILINEAR3D_CPU_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("upsample_trilinear_3d")                                         \
      .SetCreateFn<UpsampleTrilinear3DCPUKernel<dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                   \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("upsample_trilinear_3d_grad")                                    \
      .SetCreateFn<UpsampleTrilinearGrad3DCPUKernel<dtype>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                   \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_UPSAMPTRILINEAR3D_CPU_KERNEL(float)
REGISTER_UPSAMPTRILINEAR3D_CPU_KERNEL(double)

}  // namespace oneflow
