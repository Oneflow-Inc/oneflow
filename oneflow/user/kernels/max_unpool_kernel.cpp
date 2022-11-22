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
#include "fmt/core.h"
#include "oneflow/core/common/bfloat16.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/user/kernels/max_unpool_kernel_util.h"

namespace oneflow {
namespace {

template<typename T, typename IDX, typename F>
void MaxUnpoolNdForwardOrBackward(const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                  const IDX elem_num, const int64_t* indice_ptr,
                                  const int64_t hwd_size, const int64_t out_elem_num, const F& f) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX bc_idx, hwd_idx;
    index_helper.OffsetToNdIndex(num, bc_idx, hwd_idx);
    IDX idx = bc_idx * hwd_size + indice_ptr[num];
    CHECK_OR_THROW(idx >= 0 && idx < out_elem_num) << fmt::format(
        "Found an invalid max index: {}, output volumes are of size {}", idx, out_elem_num);
    f(num, idx);
  }
}

}  // namespace

template<typename T, typename IDX>
struct UnpoolKernelUtil<DeviceType::kCPU, T, IDX> {
  static void MaxUnpoolNdForward(ep::Stream* stream,
                                 const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                 const IDX elem_num, const T* src, T* dest,
                                 const int64_t* indice_ptr, const int64_t y_hwd_size,
                                 const int64_t y_elem_num) {
    MaxUnpoolNdForwardOrBackward<T>(index_helper, elem_num, indice_ptr, y_hwd_size, y_elem_num,
                                    [&](int64_t num, IDX idx) { dest[idx] = src[num]; });
  }

  static void MaxUnpoolNdBackward(ep::Stream* stream,
                                  const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                  const IDX elem_num, const T* src, T* dest,
                                  const int64_t* indice_ptr, const int64_t dy_hwd_size,
                                  const int64_t dy_elem_num) {
    MaxUnpoolNdForwardOrBackward<T>(index_helper, elem_num, indice_ptr, dy_hwd_size, dy_elem_num,
                                    [&](int64_t num, IDX idx) { dest[num] = src[idx]; });
  }
};

template<DeviceType device_type, typename T>
class MaxUnpoolNdKernel final : public user_op::OpKernel {
 public:
  MaxUnpoolNdKernel() = default;
  ~MaxUnpoolNdKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const int64_t elem_num = x->shape_view().elem_cnt();
    const T* src = x->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = y->mut_dptr<T>();

    DimVector x_vector(2);
    x_vector.at(0) = x->shape_view().At(0) * x->shape_view().At(1);

    int64_t y_hwd_size = 1;

    x_vector.at(1) = std::accumulate(x->shape_view().begin() + 2, x->shape_view().end(), 1,
                                     std::multiplies<int64_t>());
    y_hwd_size = std::accumulate(y->shape_view().begin() + 2, y->shape_view().end(), 1,
                                 std::multiplies<int64_t>());

    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
    CHECK(memset_primitive);
    memset_primitive->Launch(ctx->stream(), dest, 0,
                             y->shape_view().elem_cnt() * GetSizeOfDataType(y->data_type()));

    const int64_t y_elem_num = y->shape_view().elem_cnt();

    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 2> index_helper(x_vector.data());
      UnpoolKernelUtil<device_type, T, int32_t>::MaxUnpoolNdForward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, y_hwd_size, y_elem_num);
    } else {
      NdIndexOffsetHelper<int64_t, 2> index_helper(x_vector.data());
      UnpoolKernelUtil<device_type, T, int64_t>::MaxUnpoolNdForward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, y_hwd_size, y_elem_num);
    }
  }
};

template<DeviceType device_type, typename T>
class MaxUnpoolNdGradKernel final : public user_op::OpKernel {
 public:
  MaxUnpoolNdGradKernel() = default;
  ~MaxUnpoolNdGradKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t elem_num = dx->shape_view().elem_cnt();
    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();

    DimVector dx_vector(2);
    dx_vector.at(0) = dx->shape_view().At(0) * dx->shape_view().At(1);
    int64_t dy_hwd_size = 1;

    dx_vector.at(1) = std::accumulate(dx->shape_view().begin() + 2, dx->shape_view().end(), 1,
                                      std::multiplies<int64_t>());
    dy_hwd_size = std::accumulate(dy->shape_view().begin() + 2, dy->shape_view().end(), 1,
                                  std::multiplies<int64_t>());

    const int64_t dy_elem_num = dy->shape_view().elem_cnt();

    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 2> index_helper(dx_vector.data());
      UnpoolKernelUtil<device_type, T, int32_t>::MaxUnpoolNdBackward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, dy_hwd_size, dy_elem_num);
    } else {
      NdIndexOffsetHelper<int64_t, 2> index_helper(dx_vector.data());
      UnpoolKernelUtil<device_type, T, int64_t>::MaxUnpoolNdBackward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, dy_hwd_size, dy_elem_num);
    }
  };
};

#define REGISTER_UNPOOL_KERNELS(device, dtype)                                          \
  REGISTER_USER_KERNEL("max_unpool_1d")                                                 \
      .SetCreateFn<MaxUnpoolNdKernel<device, dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_unpool_2d")                                                 \
      .SetCreateFn<MaxUnpoolNdKernel<device, dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_unpool_3d")                                                 \
      .SetCreateFn<MaxUnpoolNdKernel<device, dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_unpool_1d_grad")                                            \
      .SetCreateFn<MaxUnpoolNdGradKernel<device, dtype>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_unpool_2d_grad")                                            \
      .SetCreateFn<MaxUnpoolNdGradKernel<device, dtype>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_unpool_3d_grad")                                            \
      .SetCreateFn<MaxUnpoolNdGradKernel<device, dtype>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNPOOL_KERNEL_UTIL, (DeviceType::kCPU),
                                 UNPOOL_DATA_TYPE_CPU_SEQ, UNPOOL_IDX_DATA_TYPE_SEQ);

#define REGISTER_UNPOOL_WITH_DEVICE(device) \
  REGISTER_UNPOOL_KERNELS(device, int32_t)  \
  REGISTER_UNPOOL_KERNELS(device, int64_t)  \
  REGISTER_UNPOOL_KERNELS(device, float)    \
  REGISTER_UNPOOL_KERNELS(device, double)

REGISTER_UNPOOL_WITH_DEVICE(DeviceType::kCPU)
REGISTER_UNPOOL_KERNELS(DeviceType::kCPU, float16)
REGISTER_UNPOOL_KERNELS(DeviceType::kCPU, bfloat16)

#ifdef WITH_CUDA
REGISTER_UNPOOL_WITH_DEVICE(DeviceType::kCUDA)
REGISTER_UNPOOL_KERNELS(DeviceType::kCUDA, half)
#if CUDA_VERSION >= 11000
REGISTER_UNPOOL_KERNELS(DeviceType::kCUDA, nv_bfloat16)
#endif  // CUDA_VERSION >= 11000
#endif  // WITH_CUDA

}  // namespace oneflow
