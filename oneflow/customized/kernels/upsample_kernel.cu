#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

template<typename T, bool align_corners>
__global__ void UpsampleNearestForward(const int64_t elem_cnt, const T* in_dptr,
                                       NdIndexOffsetHelper<int64_t, 4> in_helper,
                                       NdIndexOffsetHelper<int64_t, 4> out_helper,
                                       const int64_t in_height, const int64_t in_width,
                                       const float scale_h, const float scale_w, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    out_helper.OffsetToNdIndex(index, n, c, h, w);

    const int64_t in_h = min((align_corners) ? static_cast<int64_t>(roundf(h * scale_h))
                                             : static_cast<int64_t>(floorf(h * scale_h)),
                             in_height - 1);
    const int64_t in_w = min((align_corners) ? static_cast<int64_t>(roundf(w * scale_w))
                                             : static_cast<int64_t>(floorf(w * scale_w)),
                             in_width - 1);
    out_dptr[index] = in_dptr[in_helper.NdIndexToOffset(n, c, in_h, in_w)];
  }
}

template<typename T, bool align_corners>
__global__ void UpsampleNearestBackward(const int64_t elem_cnt, const T* dy_dptr,
                                        NdIndexOffsetHelper<int64_t, 4> dy_helper,
                                        NdIndexOffsetHelper<int64_t, 4> dx_helper,
                                        const int64_t dx_height, const int64_t dx_width,
                                        const float scale_h, const float scale_w, T* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    dy_helper.OffsetToNdIndex(index, n, c, h, w);
    const int64_t in_h = min((align_corners) ? static_cast<int64_t>(roundf(h * scale_h))
                                             : static_cast<int64_t>(floorf(h * scale_h)),
                             dx_height - 1);
    const int64_t in_w = min((align_corners) ? static_cast<int64_t>(roundf(w * scale_w))
                                             : static_cast<int64_t>(floorf(w * scale_w)),
                             dx_width - 1);
    atomicAdd(dx_dptr + dx_helper.NdIndexToOffset(n, c, in_h, in_w), dy_dptr[index]);
  }
}

template<typename T>
__global__ void UpsampleBilinearForward(const int64_t elem_cnt, const T* in_dptr,
                                        NdIndexOffsetHelper<int64_t, 4> in_helper,
                                        NdIndexOffsetHelper<int64_t, 4> out_helper,
                                        const int64_t in_height, const int64_t in_width,
                                        const float scale_h, const float scale_w, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    out_helper.OffsetToNdIndex(index, n, c, h, w);
    const float in_h = (static_cast<float>(h) + 0.5f) * scale_h - 0.5f;
    const int64_t top_h_index = in_h > 0.0 ? floorf(in_h) : 0;
    const int64_t bottom_h_index = (in_h < in_height - 1) ? ceilf(in_h) : in_height - 1;
    const float h_lerp = in_h - top_h_index;
    const float in_w = (static_cast<float>(w) + 0.5f) * scale_w - 0.5f;
    const int64_t left_w_index = in_w > 0.0 ? floorf(in_w) : 0;
    const int64_t right_w_index = (in_w < in_width - 1) ? ceilf(in_w) : in_width - 1;
    const float w_lerp = in_w - left_w_index;
    const int64_t top_offset = in_helper.NdIndexToOffset(n, c, top_h_index, 0);
    const float top_left = in_dptr[top_offset + left_w_index];
    const float top_right = in_dptr[top_offset + right_w_index];
    const int64_t bottom_offset = in_helper.NdIndexToOffset(n, c, bottom_h_index, 0);
    const float bottom_left = in_dptr[bottom_offset + left_w_index];
    const float bottom_right = in_dptr[bottom_offset + right_w_index];
    const float top = top_left + (top_right - top_left) * w_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * w_lerp;
    out_dptr[index] = top + (bottom - top) * h_lerp;
  }
}

template<typename T>
__global__ void UpsampleBilinearBackward(const int64_t elem_cnt, const T* dy_dptr,
                                         NdIndexOffsetHelper<int64_t, 4> dy_helper,
                                         NdIndexOffsetHelper<int64_t, 4> dx_helper,
                                         const int64_t dx_height, const int64_t dx_width,
                                         const float scale_h, const float scale_w, T* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    dy_helper.OffsetToNdIndex(index, n, c, h, w);
    const float original_h = (static_cast<float>(h) + 0.5f) * scale_h - 0.5f;
    const int64_t top_h_index = original_h > 0.0 ? floorf(original_h) : 0;
    const int64_t bottom_h_index = (original_h < dx_height - 1) ? ceilf(original_h) : dx_height - 1;
    const float h_lerp = original_h - floorf(original_h);
    const float original_w = (static_cast<float>(w) + 0.5f) * scale_w - 0.5f;
    const int64_t left_w_index = original_w > 0.0 ? floorf(original_w) : 0;
    const int64_t right_w_index = (original_w < dx_width - 1) ? ceilf(original_w) : dx_width - 1;
    const float w_lerp = original_w - floorf(original_w);
    const int64_t top_offset = dx_helper.NdIndexToOffset(n, c, top_h_index, 0);
    const float dtop = (1 - h_lerp) * dy_dptr[index];
    atomicAdd(dx_dptr + top_offset + left_w_index, static_cast<T>((1 - w_lerp) * dtop));
    atomicAdd(dx_dptr + top_offset + right_w_index, static_cast<T>(w_lerp * dtop));
    const int64_t bottom_offset = dx_helper.NdIndexToOffset(n, c, bottom_h_index, 0);
    const float dbottom = h_lerp * dy_dptr[index];
    atomicAdd(dx_dptr + bottom_offset + left_w_index, static_cast<T>((1 - w_lerp) * dbottom));
    atomicAdd(dx_dptr + bottom_offset + right_w_index, static_cast<T>(w_lerp * dbottom));
  }
}

}  // namespace

template<typename T>
class UpsampleNearestGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearestGPUKernel() = default;
  ~UpsampleNearestGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const int64_t elem_cnt = y_blob->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> in_helper(x_blob->shape().At(0), x_blob->shape().At(1),
                                              x_blob->shape().At(2), x_blob->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> out_helper(y_blob->shape().At(0), y_blob->shape().At(1),
                                               y_blob->shape().At(2), y_blob->shape().At(3));

    RUN_CUDA_KERNEL((UpsampleNearestForward<T, false>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    x_blob->dptr<T>(), in_helper, out_helper, x_blob->shape().At(2),
                    x_blob->shape().At(3), 1.f / height_scale, 1.f / width_scale,
                    y_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleNearestGradGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearestGradGPUKernel() = default;
  ~UpsampleNearestGradGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_blob == nullptr) { return; }
    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx_blob->mut_dptr<T>(), 0,
                             dx_blob->shape().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> dy_helper(dy_blob->shape().At(0), dy_blob->shape().At(1),
                                              dy_blob->shape().At(2), dy_blob->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> dx_helper(dx_blob->shape().At(0), dx_blob->shape().At(1),
                                              dx_blob->shape().At(2), dx_blob->shape().At(3));
    RUN_CUDA_KERNEL((UpsampleNearestBackward<T, false>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    dy_blob->dptr<T>(), dy_helper, dx_helper, dx_blob->shape().At(2),
                    dx_blob->shape().At(3), 1.f / height_scale, 1.f / width_scale,
                    dx_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_NEAREST_GPU_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("upsample")                                                       \
      .SetCreateFn<UpsampleNearestGPUKernel<dtype>>()                                    \
      .SetIsMatchedHob(                                                                  \
          (user_op::HobDeviceType() == DeviceType::kGPU)                                 \
          & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)                  \
          & (user_op::HobAttr<std::string>("interpolation") == std::string("nearest"))); \
  REGISTER_USER_KERNEL("upsample_grad")                                                  \
      .SetCreateFn<UpsampleNearestGradGPUKernel<dtype>>()                                \
      .SetIsMatchedHob(                                                                  \
          (user_op::HobDeviceType() == DeviceType::kGPU)                                 \
          & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)                 \
          & (user_op::HobAttr<std::string>("interpolation") == std::string("nearest")));

REGISTER_UPSAMPLE_NEAREST_GPU_KERNEL(float)
REGISTER_UPSAMPLE_NEAREST_GPU_KERNEL(double)

template<typename T>
class UpsampleBilinearGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBilinearGPUKernel() = default;
  ~UpsampleBilinearGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const int64_t elem_cnt = y_blob->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> in_helper(x_blob->shape().At(0), x_blob->shape().At(1),
                                              x_blob->shape().At(2), x_blob->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> out_helper(y_blob->shape().At(0), y_blob->shape().At(1),
                                               y_blob->shape().At(2), y_blob->shape().At(3));

    RUN_CUDA_KERNEL((UpsampleBilinearForward<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    x_blob->dptr<T>(), in_helper, out_helper, x_blob->shape().At(2),
                    x_blob->shape().At(3), 1.f / height_scale, 1.f / width_scale,
                    y_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleBilinearGradGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBilinearGradGPUKernel() = default;
  ~UpsampleBilinearGradGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_blob == nullptr) { return; }
    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx_blob->mut_dptr<T>(), 0,
                             dx_blob->shape().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> dy_helper(dy_blob->shape().At(0), dy_blob->shape().At(1),
                                              dy_blob->shape().At(2), dy_blob->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> dx_helper(dx_blob->shape().At(0), dx_blob->shape().At(1),
                                              dx_blob->shape().At(2), dx_blob->shape().At(3));

    RUN_CUDA_KERNEL((UpsampleBilinearBackward<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    dy_blob->dptr<T>(), dy_helper, dx_helper, dx_blob->shape().At(2),
                    dx_blob->shape().At(3), 1.f / height_scale, 1.f / width_scale,
                    dx_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_BILINEAR_GPU_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("upsample")                                                        \
      .SetCreateFn<UpsampleBilinearGPUKernel<dtype>>()                                    \
      .SetIsMatchedHob(                                                                   \
          (user_op::HobDeviceType() == DeviceType::kGPU)                                  \
          & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)                   \
          & (user_op::HobAttr<std::string>("interpolation") == std::string("bilinear"))); \
  REGISTER_USER_KERNEL("upsample_grad")                                                   \
      .SetCreateFn<UpsampleBilinearGradGPUKernel<dtype>>()                                \
      .SetIsMatchedHob(                                                                   \
          (user_op::HobDeviceType() == DeviceType::kGPU)                                  \
          & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)                  \
          & (user_op::HobAttr<std::string>("interpolation") == std::string("bilinear")));

REGISTER_UPSAMPLE_BILINEAR_GPU_KERNEL(float)
REGISTER_UPSAMPLE_BILINEAR_GPU_KERNEL(double)

}  // namespace oneflow
