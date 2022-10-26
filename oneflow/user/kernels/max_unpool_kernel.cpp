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
#include "oneflow/user/kernels/max_unpool_kernel_util.h"

namespace oneflow {

template<typename T, typename IDX>
struct UnpoolKernelUtil<DeviceType::kCPU, T, IDX> {
  static void MaxUnpoolNdForward(ep::Stream* stream,
                                 const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                 const IDX elem_num, const T* src, T* dest,
                                 const int64_t* indice_ptr, const int64_t y_hwd_size) {
    XPU_1D_KERNEL_LOOP(num, elem_num) {
      IDX bc_idx, hwd_idx;
      index_helper.OffsetToNdIndex(num, bc_idx, hwd_idx);
      IDX dest_idx = bc_idx * y_hwd_size + indice_ptr[num];
      dest[dest_idx] = src[num];
    }
  }

  static void MaxUnpoolNdBackward(ep::Stream* stream,
                                  const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                  const IDX elem_num, const T* src, T* dest,
                                  const int64_t* indice_ptr, const int64_t dx_hwd_size) {
    XPU_1D_KERNEL_LOOP(num, elem_num) {
      IDX bc_idx, hwd_idx;
      index_helper.OffsetToNdIndex(num, bc_idx, hwd_idx);
      IDX src_idx = bc_idx * dx_hwd_size + indice_ptr[num];
      dest[num] = src[src_idx];
    }
  }
};

template<DeviceType device_type, typename T, int NDIMS>
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
    if (NDIMS == 1) {
      x_vector.at(1) = x->shape_view().At(2);
      y_hwd_size = y->shape_view().At(2);
    } else if (NDIMS == 2) {
      x_vector.at(1) = x->shape_view().At(2) * x->shape_view().At(3);
      y_hwd_size = y->shape_view().At(2) * y->shape_view().At(3);
    } else if (NDIMS == 3) {
      x_vector.at(1) = x->shape_view().At(2) * x->shape_view().At(3) * x->shape_view().At(4);
      y_hwd_size = y->shape_view().At(2) * y->shape_view().At(3) * y->shape_view().At(4);
    }

    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
    CHECK(memset_primitive);
    memset_primitive->Launch(ctx->stream(), dest, 0, y->shape_view().elem_cnt());

    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 2> index_helper(x_vector.data());
      UnpoolKernelUtil<device_type, T, int32_t>::MaxUnpoolNdForward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, y_hwd_size);
    } else {
      NdIndexOffsetHelper<int64_t, 2> index_helper(x_vector.data());
      UnpoolKernelUtil<device_type, T, int64_t>::MaxUnpoolNdForward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, y_hwd_size);
    }
  }
};

template<DeviceType device_type, typename T, int NDIMS>
class MaxUnpoolNdGradKernel final : public user_op::OpKernel {
 public:
  MaxUnpoolNdGradKernel() = default;
  ~MaxUnpoolNdGradKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t elem_num = dx->shape_view().elem_cnt();
    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();
    DimVector dy_vector(2);
    dy_vector.at(0) = dy->shape_view().At(0) * dy->shape_view().At(1);

    int64_t dx_hwd_size = 1;
    if (NDIMS == 1) {
      dy_vector.at(1) = dy->shape_view().At(2);
      dx_hwd_size = dx->shape_view().At(2);
    } else if (NDIMS == 2) {
      dy_vector.at(1) = dy->shape_view().At(2) * dy->shape_view().At(3);
      dx_hwd_size = dx->shape_view().At(2) * dx->shape_view().At(3);
    } else if (NDIMS == 3) {
      dy_vector.at(1) = dy->shape_view().At(2) * dy->shape_view().At(3) * dy->shape_view().At(4);
      dx_hwd_size = dx->shape_view().At(2) * dx->shape_view().At(3) * dx->shape_view().At(4);
    }

    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
    CHECK(memset_primitive);
    memset_primitive->Launch(ctx->stream(), dest, 0, dx->shape_view().elem_cnt());

    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 2> index_helper(dy_vector.data());
      UnpoolKernelUtil<device_type, T, int32_t>::MaxUnpoolNdBackward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, dx_hwd_size);
    } else {
      NdIndexOffsetHelper<int64_t, 2> index_helper(dy_vector.data());
      UnpoolKernelUtil<device_type, T, int64_t>::MaxUnpoolNdBackward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, dx_hwd_size);
    }
  };
};

#define REGISTER_UNPOOL_KERNELS(device, dtype)                                          \
  REGISTER_USER_KERNEL("max_unpool_1d")                                                 \
      .SetCreateFn<MaxUnpoolNdKernel<device, dtype, 1>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_unpool_2d")                                                 \
      .SetCreateFn<MaxUnpoolNdKernel<device, dtype, 2>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_unpool_3d")                                                 \
      .SetCreateFn<MaxUnpoolNdKernel<device, dtype, 3>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_unpool_1d_grad")                                            \
      .SetCreateFn<MaxUnpoolNdGradKernel<device, dtype, 1>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_unpool_2d_grad")                                            \
      .SetCreateFn<MaxUnpoolNdGradKernel<device, dtype, 2>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_unpool_3d_grad")                                            \
      .SetCreateFn<MaxUnpoolNdGradKernel<device, dtype, 3>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

#define REGISTER_UNPOOL_WITH_DEVICE(device) \
  REGISTER_UNPOOL_KERNELS(device, int32_t)  \
  REGISTER_UNPOOL_KERNELS(device, float)    \
  REGISTER_UNPOOL_KERNELS(device, double)

REGISTER_UNPOOL_WITH_DEVICE(DeviceType::kCPU)

#ifdef WITH_CUDA
REGISTER_UNPOOL_WITH_DEVICE(DeviceType::kCUDA)
REGISTER_UNPOOL_KERNELS(DeviceType::kCUDA, half)
#endif

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNPOOL_KERNEL_UTIL, (DeviceType::kCPU),
                                 UNPOOL_DATA_TYPE_CPU_SEQ, UNPOOL_IDX_DATA_TYPE_SEQ);

}  // namespace oneflow
